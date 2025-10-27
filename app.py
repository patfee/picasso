import os
import csv
import io
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from scipy.interpolate import RegularGridInterpolator
import shapely.geometry as sgeom
import alphashape

# =========================
# Config
# =========================
DATA_DIR = os.getenv("DATA_DIR", "./data")
HEIGHT_CSV = os.getenv("HEIGHT_CSV", "height.csv")
OUTREACH_CSV = os.getenv("OUTREACH_CSV", "outreach.csv")
PEDESTAL_HEIGHT_M = float(os.getenv("PEDESTAL_HEIGHT_M", "6"))
# Put your logo file at ./assets/dcn_logo.svg (Dash serves /assets/* automatically)
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")

# Display/perf caps
MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 8000


# =========================
# CSV Loading
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
# Interpolation & Geometry
# =========================
def build_interpolators(height_df: pd.DataFrame, outreach_df: pd.DataFrame):
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.loc[fold_angles, main_angles]

    height_itp = RegularGridInterpolator((fold_angles, main_angles), height_df.values, bounds_error=False, fill_value=None)
    outre_itp = RegularGridInterpolator((fold_angles, main_angles), outreach_df.values, bounds_error=False, fill_value=None)
    return fold_angles, main_angles, height_itp, outre_itp


def resample_grid(fold_angles, main_angles, d_fold: float, d_main: float):
    fmin, fmax = float(np.min(fold_angles)), float(np.max(fold_angles))
    mmin, mmax = float(np.min(main_angles)), float(np.max(main_angles))
    d_fold, d_main = max(0.1, d_fold), max(0.1, d_main)

    fnew = np.arange(fmin, fmax + 1e-9, d_fold)
    mnew = np.arange(mmin, mmax + 1e-9, d_main)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])
    return fnew, mnew, F, M, pts


def _sample_points(arr: np.ndarray, max_n: int) -> np.ndarray:
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]


def compute_boundary_curve(xy_points: np.ndarray, prefer_concave: bool = True):
    if xy_points is None:
        return None
    pts = np.asarray(xy_points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None

    # remove duplicates
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None

    # detect near-collinearity
    centered = pts - pts.mean(axis=0)
    try:
        s = np.linalg.svd(centered, compute_uv=False)
        near_1d = (len(s) < 2) or (s[1] <= 1e-8 * s[0])
    except Exception:
        near_1d = True

    if near_1d:
        try:
            hull = sgeom.MultiPoint(pts).convex_hull
            if isinstance(hull, sgeom.Polygon):
                x, y = hull.exterior.coords.xy
                return np.column_stack([np.asarray(x), np.asarray(y)])
            elif isinstance(hull, sgeom.LineString):
                x, y = hull.coords.xy
                return np.column_stack([np.asarray(x), np.asarray(y)])
            return None
        except Exception:
            return None

    poly = None
    if prefer_concave:
        try:
            pts_cap = _sample_points(pts, MAX_POINTS_FOR_ENVELOPE)
            alpha = alphashape.optimizealpha(pts_cap)
            poly = alphashape.alphashape(pts_cap, alpha)
        except Exception:
            poly = None

    if poly is None:
        poly = sgeom.MultiPoint(pts).convex_hull

    try:
        if isinstance(poly, sgeom.MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if isinstance(poly, sgeom.Polygon):
            x, y = poly.exterior.coords.xy
            return np.column_stack([np.asarray(x), np.asarray(y)])
        if isinstance(poly, sgeom.LineString):
            x, y = poly.coords.xy
            return np.column_stack([np.asarray(x), np.asarray(y)])
    except Exception:
        pass
    return None


def make_figure(orig_xy: np.ndarray, dense_xy: np.ndarray, boundary_xy, include_pedestal: bool):
    fig = go.Figure()

    dense_for_plot = _sample_points(dense_xy, MAX_POINTS_FOR_DISPLAY)

    if dense_for_plot.size:
        fig.add_trace(go.Scatter(
            x=dense_for_plot[:, 0], y=dense_for_plot[:, 1],
            mode="markers",
            marker=dict(size=4, symbol="diamond", opacity=0.5),
            name="Interpolated points",
        ))
    if orig_xy.size:
        fig.add_trace(go.Scatter(
            x=orig_xy[:, 0],
            y=orig_xy[:, 1] + (PEDESTAL_HEIGHT_M if include_pedestal else 0.0),
            mode="markers",
            marker=dict(size=8),
            name="Original matrix points",
        ))
    if boundary_xy is not None and len(boundary_xy) > 2:
        fig.add_trace(go.Scatter(
            x=boundary_xy[:, 0], y=boundary_xy[:, 1],
            mode="lines", line=dict(width=3),
            name="Envelope",
        ))

    fig.update_layout(
        # remove plot title (we have a page header) to avoid legend overlap
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head height [m]" if include_pedestal else "Jib head above pedestal flange [m]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="keep",
    )
    return fig


# ---------- helper to build tidy slider marks (min / mid / max only)
def _three_marks(vmin: float, vmax: float, unit: str):
    mid = (vmin + vmax) / 2.0
    style = {"fontSize": "12px", "color": "#555"}
    return {
        round(vmin, 2): {"label": f"{vmin:g}{unit}", "style": style},
        round(mid, 2): {"label": f"{mid:g}{unit}", "style": style},
        round(vmax, 2): {"label": f"{vmax:g}{unit}", "style": style},
    }


# =========================
# App Factory (multi-page)
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                    title="DCN Picasso DSV Engineering Data")

    # --- Load data ---
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    # --- Slider mark domains (for readability; marks are static)
    step_min, step_max = 0.1, 10.0
    step_marks = _three_marks(step_min, step_max, "°")

    # --- Shared Stores ---
    stores = html.Div([
        dcc.Store(id="orig-xy", data=orig_xy.tolist()),
        dcc.Store(id="fold-angles", data=fold_angles.tolist()),
        dcc.Store(id="main-angles", data=main_angles.tolist()),
    ])

    # --- Sidebar Navigation ---
    sidebar = html.Div([
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
        html.Br(),
        dcc.Link("Test", href="/test", className="nav-link"),
    ], style={
        "position": "sticky", "top": "0px", "height": "100vh",
        "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa",
    })

    # --- Header ---
    header = html.Div([
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(src=LOGO_URL,
                 loading="lazy",
                 style={"height": "42px", "objectFit": "contain", "position": "absolute",
                        "right": "16px", "top": "12px"},
                 alt="DCN Diving logo"),
    ], style={"position": "relative", "padding": "12px 16px 4px 16px"})

    # --- Envelope Page (controls + graph in two columns) ---
    envelope_controls = html.Div([
        html.Label("Step between Main-jib matrix angles (°)", style={"fontWeight": 600}),
        dcc.Slider(
            id="slider-main-step", min=step_min, max=step_max, step=0.1, value=1.0,
            marks=step_marks, included=False, updatemode="mouseup",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="main-range", style={"marginTop": "6px", "fontFamily": "monospace", "color": "#444"}),

        html.Label("Step between Folding-jib matrix angles (°)", style={"fontWeight": 600, "marginTop": "10px"}),
        dcc.Slider(
            id="slider-fold-step", min=step_min, max=step_max, step=0.1, value=1.0,
            marks=step_marks, included=False, updatemode="mouseup",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(id="fold-range", style={"marginTop": "6px", "fontFamily": "monospace", "color": "#444"}),

        dcc.Checklist(
            id="pedestal-toggle",
            options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
            value=["on"], style={"marginTop": "12px"}
        ),

        html.Label("Envelope type", style={"fontWeight": 600, "marginTop": "6px"}),
        dcc.RadioItems(
            id="envelope-type",
            options=[{"label": "Concave (alpha shape)", "value": "concave"},
                     {"label": "Convex hull", "value": "convex"}],
            value="convex",
            style={"marginTop": "5px"}
        ),

        html.Div([
            html.Button("Download interpolated CSV", id="btn-download", n_clicks=0,
                        style={"marginTop": "10px", "width": "100%"}),
            dcc.Download(id="download-data"),
        ], style={"position": "sticky", "bottom": 0, "background": "#fafafa", "padding": "8px 0"}),

    ], style={
        "flex": "0 0 340px",
        "overflowY": "auto",
        "height": "78vh",
        "paddingRight": "12px",
        "borderRight": "1px solid #eee"
    })

    envelope_graph = html.Div(
        dcc.Graph(id="graph", style={"height": "78vh"}),
        style={"flex": "1 1 auto", "paddingLeft": "16px"}
    )

    envelope_page = html.Div(
        [envelope_controls, envelope_graph],
        style={"display": "flex", "padding": "16px", "gap": "10px"}
    )

    test_page = html.Div([html.H3("Test Page"), html.P("Placeholder for future content.")], style={"padding": "16px"})
    content = html.Div(id="page-content", style={"width": "100%"})

    app.layout = html.Div([
        dcc.Location(id="url"), stores, header,
        html.Div([
            html.Div(sidebar, style={"width": "260px", "flex": "0 0 260px"}),
            html.Div(content, style={"flex": "1 1 auto"}),
        ], style={"display": "flex", "gap": "0px"}),
    ])

    # --- Routing ---
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def route(pathname: str):
        if pathname in ("/", "/envelope", None):
            return envelope_page
        elif pathname == "/test":
            return test_page
        return html.Div([html.H3("404"), html.P(f"Path not found: {pathname}")], style={"padding": "16px"})

    # --- Show angle ranges ---
    @app.callback(
        Output("main-range", "children"),
        Output("fold-range", "children"),
        Input("main-angles", "data"),
        Input("fold-angles", "data"),
    )
    def show_ranges(main_angles_store, fold_angles_store):
        main_arr, fold_arr = np.array(main_angles_store, float), np.array(fold_angles_store, float)
        return (f"Main-jib angle range: {main_arr.min():.2f}° – {main_arr.max():.2f}°",
                f"Folding-jib angle range: {fold_arr.min():.2f}° – {fold_arr.max():.2f}°")

    # --- Graph update ---
    @app.callback(
        Output("graph", "figure"),
        Input("slider-main-step", "value"),
        Input("slider-fold-step", "value"),
        Input("pedestal-toggle", "value"),
        Input("envelope-type", "value"),
        State("orig-xy", "data"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
    )
    def update_figure(main_step, fold_step, pedestal_value, envelope_value,
                      orig_xy_store, fold_angles_store, main_angles_store):
        include_pedestal = "on" in (pedestal_value or [])
        prefer_concave = (envelope_value == "concave")

        _, _, _, _, pts = resample_grid(np.array(fold_angles_store), np.array(main_angles_store),
                                        d_fold=fold_step, d_main=main_step)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M

        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

        envelope_xy = compute_boundary_curve(
            _sample_points(dense_xy, MAX_POINTS_FOR_ENVELOPE),
            prefer_concave=prefer_concave
        )

        return make_figure(np.array(orig_xy_store), dense_xy, envelope_xy, include_pedestal)

    # --- Download ---
    @app.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        State("slider-main-step", "value"),
        State("slider-fold-step", "value"),
        State("pedestal-toggle", "value"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
        prevent_initial_call=True,
    )
    def download_interpolated(n_clicks, main_step, fold_step, pedestal_value, fold_angles_store, main_angles_store):
        if not n_clicks:
            return dash.no_update
        include_pedestal = "on" in (pedestal_value or [])
        _, _, _, _, pts = resample_grid(np.array(fold_angles_store), np.array(main_angles_store),
                                        d_fold=fold_step, d_main=main_step)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M
        df_out = pd.DataFrame({
            "FoldingJib_deg": pts[:, 0],
            "MainJib_deg": pts[:, 1],
            "Outreach_m": R_dense,
            "Height_m": H_dense,
            "PedestalIncluded": include_pedestal,
        }).dropna()
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        return dcc.send_bytes(lambda b: b.write(csv_bytes), filename="interpolated_envelope.csv")

    return app, server


# =========================
# Entrypoint
# =========================
app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
