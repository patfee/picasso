import os
import csv
import io
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go

from scipy.interpolate import RegularGridInterpolator, griddata
import shapely.geometry as sgeom
import alphashape


# =========================
# Config
# =========================
DATA_DIR = os.getenv("DATA_DIR", "./data")
HEIGHT_CSV = os.getenv("HEIGHT_CSV", "height.csv")
OUTREACH_CSV = os.getenv("OUTREACH_CSV", "outreach.csv")
LOAD_CSV = os.getenv("LOAD_CSV", "Harbour_Cdyn115.csv")
PEDESTAL_HEIGHT_M = float(os.getenv("PEDESTAL_HEIGHT_M", "6"))
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")
ASSETS_DIR = os.getenv("ASSETS_DIR", "./assets")

MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN = 1200
KNN_K = 8


# =========================
# CSV Helpers
# =========================
def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        return ";"


def _clean_angles(vals):
    cleaned = (
        pd.Series(vals)
        .astype(str)
        .str.replace("°", "", regex=False)
        .str.replace("deg", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_matrix_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    delim = _sniff_delimiter(raw[:4096])
    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        df = pd.read_csv(io.StringIO(raw), sep=delim, engine="python")
        first_col = df.columns[0]
        if (
            first_col.lower().strip()
            in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
            or df[first_col]
            .dropna()
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.match(r"^\s*-?\d+(\.\d+)?\s*$")
            .all()
        ):
            df = df.set_index(first_col)
    df.columns = _clean_angles(df.columns)
    df.index = _clean_angles(df.index)
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.sort_index().sort_index(axis=1)
    if df.empty:
        raise ValueError(f"{path} parsed to empty DataFrame.")
    return df


# =========================
# Geometry & interpolation
# =========================
def build_interpolators(height_df: pd.DataFrame, outreach_df: pd.DataFrame):
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.loc[fold_angles, main_angles]
    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles), height_df.values, bounds_error=False, fill_value=None
    )
    outre_itp = RegularGridInterpolator(
        (fold_angles, main_angles), outreach_df.values, bounds_error=False, fill_value=None
    )
    return fold_angles, main_angles, height_itp, outre_itp


def resample_grid_by_factors(fold_angles, main_angles, fold_factor, main_factor):
    def expand(arr, f):
        arr = np.unique(np.asarray(arr, float))
        arr.sort()
        if len(arr) < 2 or int(f) <= 1:
            return arr
        out = [arr[0]]
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            seg = np.linspace(a, b, int(f) + 1, endpoint=True)[1:]
            out.extend(seg.tolist())
        return np.array(out, float)

    fnew = expand(fold_angles, fold_factor)
    mnew = expand(main_angles, main_factor)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])
    return fnew, mnew, F, M, pts


def _sample_points(arr, max_n):
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]


def _estimate_alpha(pts, k=KNN_K):
    pts = _sample_points(pts, MAX_POINTS_FOR_KNN)
    if len(pts) < k + 2:
        return 1.0
    A = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((A**2).sum(axis=2))
    D.sort(axis=1)
    dk = np.median(D[:, k])
    if not np.isfinite(dk) or dk <= 1e-9:
        return 1.0
    return 1.0 / (1.8 * dk)


def compute_boundary_curve(xy_points, prefer_concave=True):
    pts = np.asarray(xy_points, float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None
    if prefer_concave:
        try:
            pts_cap = _sample_points(pts, MAX_POINTS_FOR_ENVELOPE)
            alpha = _estimate_alpha(pts_cap)
            poly = alphashape.alphashape(pts_cap, alpha)
        except Exception:
            poly = None
    else:
        poly = None
    if poly is None:
        poly = sgeom.MultiPoint(pts).convex_hull
    if isinstance(poly, sgeom.MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if isinstance(poly, sgeom.Polygon):
        x, y = poly.exterior.coords.xy
        return np.column_stack([x, y])
    if isinstance(poly, sgeom.LineString):
        x, y = poly.coords.xy
        return np.column_stack([x, y])
    return None


# =========================
# App Factory
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                    title="DCN Picasso DSV Engineering Data")

    # --- Load data ---
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    load_df = load_matrix_csv(os.path.join(DATA_DIR, LOAD_CSV))
    load_df = load_df.reindex(index=height_df.index, columns=height_df.columns)

    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])
    F0, M0 = np.meshgrid(fold_angles, main_angles, indexing="ij")
    orig_custom = np.column_stack([F0.ravel(), M0.ravel()])

    stores = html.Div([
        dcc.Store(id="orig-xy", data=orig_xy.tolist()),
        dcc.Store(id="orig-angles", data=orig_custom.tolist()),
        dcc.Store(id="fold-angles", data=fold_angles.tolist()),
        dcc.Store(id="main-angles", data=main_angles.tolist()),
    ])

    # --- Sidebar ---
    sidebar = html.Div([
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
        html.Br(),
        dcc.Link("Harbour Mode", href="/harbour", className="nav-link"),
        html.Br(),
        dcc.Link("Test", href="/test", className="nav-link"),
        html.Br(),
        dcc.Link("TTS Data", href="/tts", className="nav-link"),
    ], style={
        "position": "sticky", "top": "0px", "height": "100vh",
        "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa",
    })

    # --- Header ---
    header = html.Div([
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(src=LOGO_URL,
                 style={"height": "42px", "objectFit": "contain", "position": "absolute",
                        "right": "16px", "top": "12px"},
                 alt="DCN Diving logo"),
    ], style={"position": "relative", "padding": "12px 16px 4px 16px"})

    # --- Envelope page (existing) ---
    def factor_dropdown(id_, label, options, value):
        return html.Div([
            html.Label(label, style={"fontWeight": 600}),
            dcc.Dropdown(
                id=id_,
                options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                         for o in options],
                value=value, clearable=False, style={"marginBottom": "8px"}
            ),
        ], style={"marginBottom": "8px"})

    envelope_controls = html.Div([
        factor_dropdown("main-factor", "Main-jib subdivision", [1, 2, 4, 8], 1),
        factor_dropdown("fold-factor", "Folding-jib subdivision", [1, 2, 4, 8, 16], 1),
        html.Div(id="main-range", style={"fontFamily": "monospace"}),
        html.Div(id="fold-range", style={"fontFamily": "monospace"}),
        dcc.Checklist(
            id="pedestal-toggle",
            options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
            value=["on"]
        ),
        html.Label("Envelope type", style={"fontWeight": 600}),
        dcc.RadioItems(
            id="envelope-type",
            options=[
                {"label": "None (points only)", "value": "none"},
                {"label": "Concave (fast approx)", "value": "concave"},
                {"label": "Convex hull (fast, stable)", "value": "convex"},
            ],
            value="convex", style={"marginTop": "4px"}
        ),
        html.Div(id="envelope-help", style={"fontSize": "12px", "color": "#555"}),
        html.Div([
            html.Button("Download interpolated CSV", id="btn-download", n_clicks=0,
                        style={"marginTop": "10px", "width": "100%"}),
            dcc.Download(id="download-data"),
        ])
    ], style={"flex": "0 0 340px", "overflowY": "auto", "height": "78vh",
              "paddingRight": "12px", "borderRight": "1px solid #eee"})

    envelope_graph = html.Div([
        dcc.Graph(id="graph", style={"height": "78vh"}),
        html.Div([
            dash_table.DataTable()
        ])
    ], style={"flex": "1 1 auto", "paddingLeft": "16px"})

    envelope_page = html.Div([envelope_controls, envelope_graph],
                             style={"display": "flex", "padding": "16px", "gap": "10px"})

    test_page = html.Div([html.H3("Test Page"), html.P("Placeholder for future content.")],
                         style={"padding": "16px"})

    tts_page = html.Div([
        html.H3("TTS Data", style={"marginTop": 0}),
        html.Div([
            html.Button("Download: Picasso_TTS_140_Tons_CraneGA.pdf", id="btn-tts-ga", n_clicks=0),
            dcc.Download(id="download-tts-ga"),
        ]),
        html.Div([
            html.Button("Download: 140Tons_TTS_Crane_load_charts.pdf", id="btn-tts-charts", n_clicks=0),
            dcc.Download(id="download-tts-charts"),
        ]),
        html.Hr(),
        html.Ul([
            html.Li(html.A("Open GA (inline)", href="/assets/Picasso_TTS_140_Tons_CraneGA.pdf", target="_blank")),
            html.Li(html.A("Open Load Charts (inline)", href="/assets/140Tons_TTS_Crane_load_charts.pdf",
                           target="_blank")),
        ]),
    ], style={"padding": "16px"})

    # --- Harbour Mode page ---
    harbour_page = html.Div([
        html.H3("Harbour Mode (Cdyn 1.15)", style={"marginTop": 0}),
        dcc.Graph(id="harbour-heatmap", style={"height": "88vh"}),
    ], style={"padding": "16px"})

    content = html.Div(id="page-content", style={"width": "100%"}, children=envelope_page)

    app.layout = html.Div([
        dcc.Location(id="url"), stores, header,
        html.Div([
            html.Div(sidebar, style={"width": "260px", "flex": "0 0 260px"}),
            html.Div(content, style={"flex": "1 1 auto"}),
        ], style={"display": "flex"})
    ])

    # --- Routing ---
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def route(path):
        if path in ("/", "/envelope", None):
            return envelope_page
        elif path == "/harbour":
            return harbour_page
        elif path == "/test":
            return test_page
        elif path == "/tts":
            return tts_page
        return html.Div([html.H3("404"), html.P(f"Path not found: {path}")], style={"padding": "16px"})

    # --- Harbour contour plot ---
    @app.callback(Output("harbour-heatmap", "figure"),
                  Input("url", "pathname"), prevent_initial_call=False)
    def draw_harbour_heatmap(_):
        R = outreach_df.values.ravel()
        H = height_df.values.ravel()
        Z = load_df.values.ravel()
        mask = np.isfinite(R) & np.isfinite(H) & np.isfinite(Z)
        R, H, Z = R[mask], H[mask], Z[mask]

        r_lin = np.linspace(np.min(R), np.max(R), 200)
        h_lin = np.linspace(np.min(H), np.max(H), 200)
        RR, HH = np.meshgrid(r_lin, h_lin)
        Z_grid = griddata(np.column_stack([R, H]), Z, (RR, HH), method="linear")
        if np.isnan(Z_grid).any():
            Z_nn = griddata(np.column_stack([R, H]), Z, (RR, HH), method="nearest")
            Z_grid = np.where(np.isnan(Z_grid), Z_nn, Z_grid)

        env_xy = compute_boundary_curve(np.column_stack([R, H]), True)
        if env_xy is not None and len(env_xy) > 2:
            try:
                poly = sgeom.Polygon(env_xy)
                pts = np.column_stack([RR.ravel(), HH.ravel()])
                inside = np.array([poly.contains(sgeom.Point(xy)) for xy in pts]).reshape(RR.shape)
                Z_grid = np.where(inside, Z_grid, np.nan)
            except Exception:
                pass

        levels = dict(start=0, end=140, size=35, coloring="heatmap", showlines=False)
        colorscale = [
            [0.00, "#00b8ff"],
            [0.20, "#5ad5ff"],
            [0.40, "#20c997"],
            [0.60, "#b2df28"],
            [0.80, "#ffd84d"],
            [0.95, "#f39c12"],
            [1.00, "#c33b2b"],
        ]

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=r_lin, y=h_lin, z=Z_grid,
            contours=levels, colorscale=colorscale,
            colorbar=dict(title="Load [t]", tickmode="array", tickvals=[0, 35, 70, 105, 140]),
            hovertemplate="Outreach: %{x:.2f} m<br>Height: %{y:.2f} m<br>Load: %{z:.2f} t<extra></extra>",
            connectgaps=False
        ))
        if env_xy is not None and len(env_xy) > 2:
            fig.add_trace(go.Scatter(
                x=env_xy[:, 0], y=env_xy[:, 1],
                mode="lines", line=dict(color="#ffd84d", width=3),
                name="Envelope", hoverinfo="skip"
            ))

        fig.update_layout(
            title="Cdyn=1.15",
            template="plotly_dark",
            paper_bgcolor="#0b2733", plot_bgcolor="#0b2733",
            xaxis=dict(title="Outreach [m]", showgrid=True, gridcolor="#214a57", zeroline=False),
            yaxis=dict(title="Height [m]", showgrid=True, gridcolor="#214a57", zeroline=False),
            margin=dict(l=60, r=20, t=50, b=50),
            uirevision="keep", showlegend=False,
        )
        return fig

    # --- TTS Data downloads ---
    @app.callback(Output("download-tts-ga", "data"),
                  Input("btn-tts-ga", "n_clicks"), prevent_initial_call=True)
    def download_tts_ga(_):
        path = os.path.join(ASSETS_DIR, "Picasso_TTS_140_Tons_CraneGA.pdf")
        with open(path, "rb") as f:
            data = f.read()
        return dcc.send_bytes(lambda b: b.write(data), filename="Picasso_TTS_140_Tons_CraneGA.pdf")

    @app.callback(Output("download-tts-charts", "data"),
                  Input("btn-tts-charts", "n_clicks"), prevent_initial_call=True)
    def download_tts_charts(_):
        path = os.path.join(ASSETS_DIR, "140Tons_TTS_Crane_load_charts.pdf")
        with open(path, "rb") as f:
            data = f.read()
        return dcc.send_bytes(lambda b: b.write(data), filename="140Tons_TTS_Crane_load_charts.pdf")

    return app, server


# =========================
# Entrypoint
# =========================
app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
