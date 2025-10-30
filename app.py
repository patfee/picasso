import os
import csv
import io
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State, dash_table
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
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")

MAX_POINTS_FOR_DISPLAY  = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN      = 1200
KNN_K                   = 8


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
    df.index   = _clean_angles(df.index)
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

    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles), height_df.values, bounds_error=False, fill_value=None
    )
    outre_itp  = RegularGridInterpolator(
        (fold_angles, main_angles), outreach_df.values, bounds_error=False, fill_value=None
    )
    return fold_angles, main_angles, height_itp, outre_itp


def expand_angles_by_factor(sorted_angles: np.ndarray, factor: int) -> np.ndarray:
    arr = np.unique(np.asarray(sorted_angles, dtype=float))
    arr.sort()
    if len(arr) < 2 or int(factor) <= 1:
        return arr.copy()
    out = [arr[0]]
    for i in range(len(arr) - 1):
        a, b = arr[i], arr[i + 1]
        seg = np.linspace(a, b, int(factor) + 1, endpoint=True)[1:]
        out.extend(seg.tolist())
    return np.array(out, dtype=float)


def resample_grid_by_factors(fold_angles: np.ndarray, main_angles: np.ndarray,
                             fold_factor: int, main_factor: int):
    fnew = expand_angles_by_factor(fold_angles, fold_factor)
    mnew = expand_angles_by_factor(main_angles, main_factor)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])  # [Folding, Main]
    return fnew, mnew, F, M, pts


def _sample_points(arr: np.ndarray, max_n: int) -> np.ndarray:
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]


def _near_collinear(pts: np.ndarray) -> bool:
    centered = pts - pts.mean(axis=0)
    try:
        s = np.linalg.svd(centered, compute_uv=False)
        return (len(s) < 2) or (s[1] <= 1e-8 * s[0])
    except Exception:
        return True


def _estimate_alpha(pts: np.ndarray, k: int = KNN_K) -> float:
    pts = _sample_points(pts, MAX_POINTS_FOR_KNN)
    if len(pts) < k + 2:
        return 1.0
    A = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((A ** 2).sum(axis=2))
    D.sort(axis=1)
    dk = np.median(D[:, k])
    if not np.isfinite(dk) or dk <= 1e-9:
        return 1.0
    c = 1.8
    return 1.0 / (c * dk)


def compute_boundary_curve(xy_points: np.ndarray, prefer_concave: bool = True):
    if xy_points is None:
        return None
    pts = np.asarray(xy_points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None
    if _near_collinear(pts):
        try:
            hull = sgeom.MultiPoint(pts).convex_hull
            if isinstance(hull, sgeom.Polygon):
                x, y = hull.exterior.coords.xy
                return np.column_stack([x, y])
            elif isinstance(hull, sgeom.LineString):
                x, y = hull.coords.xy
                return np.column_stack([x, y])
            return None
        except Exception:
            return None
    poly = None
    if prefer_concave:
        try:
            pts_cap = _sample_points(pts, MAX_POINTS_FOR_ENVELOPE)
            alpha = _estimate_alpha(pts_cap, k=KNN_K)
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
            return np.column_stack([x, y])
        if isinstance(poly, sgeom.LineString):
            x, y = poly.coords.xy
            return np.column_stack([x, y])
    except Exception:
        pass
    return None


def make_figure(orig_xy, dense_xy, boundary_xy, include_pedestal,
                dense_custom=None, orig_custom=None):
    fig = go.Figure()
    dense_for_plot = _sample_points(dense_xy, MAX_POINTS_FOR_DISPLAY)
    dense_custom_plot = None
    if dense_custom is not None and len(dense_custom) == len(dense_xy):
        dense_custom_plot = _sample_points(dense_custom, MAX_POINTS_FOR_DISPLAY)
    pedestal_str = "yes" if include_pedestal else "no"
    hover_tmpl = (
        "Main Jib: %{customdata[1]:.2f} deg, Folding Jib: %{customdata[0]:.2f} deg<br>"
        "Outreach: %{x:.2f} m, Height: %{y:.2f} m<br>"
        f"Pedestal: {pedestal_str}<extra></extra>"
    )
    if dense_for_plot.size:
        fig.add_trace(go.Scatter(
            x=dense_for_plot[:, 0],
            y=dense_for_plot[:, 1],
            mode="markers",
            marker=dict(size=4, symbol="diamond", opacity=0.5),
            name="Interpolated points",
            customdata=dense_custom_plot,
            hovertemplate=hover_tmpl if dense_custom_plot is not None else None,
        ))
    if orig_xy.size:
        fig.add_trace(go.Scatter(
            x=orig_xy[:, 0],
            y=orig_xy[:, 1] + (PEDESTAL_HEIGHT_M if include_pedestal else 0.0),
            mode="markers",
            marker=dict(size=8),
            name="Original matrix points",
            customdata=orig_custom,
            hovertemplate=hover_tmpl if orig_custom is not None else None,
        ))
    if boundary_xy is not None and len(boundary_xy) > 2:
        fig.add_trace(go.Scatter(
            x=boundary_xy[:, 0],
            y=boundary_xy[:, 1],
            mode="lines",
            line=dict(width=3),
            name="Envelope",
        ))
    fig.update_layout(
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head height [m]" if include_pedestal else "Jib head above pedestal flange [m]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="keep",
    )
    return fig


def df_to_dash_table(df, title, table_id):
    display_df = df.copy()
    display_df.index.name = "Folding°"
    display_df = display_df.reset_index()
    columns = [{"name": str(c), "id": str(c)} for c in display_df.columns]
    data = display_df.rename(columns=str).to_dict("records")
    return html.Div([
        html.H4(title, style={"margin": "12px 0 6px 0"}),
        dash_table.DataTable(
            id=table_id,
            columns=columns,
            data=data,
            page_size=25,
            style_table={"overflowX": "auto", "border": "1px solid #eee"},
            style_cell={"padding": "6px", "fontFamily": "monospace", "fontSize": "12px"},
            style_header={"fontWeight": "600", "backgroundColor": "#f6f6f6"},
        ),
    ])


# =========================
# App Factory
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                    title="DCN Picasso DSV Engineering Data")

    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
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

    sidebar = html.Div([
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T main hoist envelope", href="/envelope"),
        html.Br(),
        dcc.Link("Test", href="/test"),
    ], style={"position": "sticky", "top": 0, "height": "100vh",
              "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa"})

    header = html.Div([
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(src=LOGO_URL, style={"height": "42px", "position": "absolute",
                                      "right": "16px", "top": "12px"}, alt="DCN logo"),
    ], style={"position": "relative", "padding": "12px 16px 4px 16px"})

    def factor_dropdown(id_, label, options, value):
        return html.Div([
            html.Label(label, style={"fontWeight": 600}),
            dcc.Dropdown(
                id=id_,
                options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o}
                         for o in options],
                value=value,
                clearable=False,
                style={"marginBottom": "8px"}
            ),
        ])

    envelope_controls = html.Div([
        factor_dropdown("main-factor", "Main-jib subdivision", [1, 2, 4, 8], 1),
        factor_dropdown("fold-factor", "Folding-jib subdivision", [1, 2, 4, 8, 16], 1),

        dcc.Checklist(
            id="pedestal-toggle",
            options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
            value=["on"], style={"marginTop": "8px"}
        ),

        html.Label("Envelope type", style={"fontWeight": 600, "marginTop": "8px"}),
        dcc.RadioItems(
            id="envelope-type",
            options=[
                {"label": "None (points only)", "value": "none"},
                {"label": "Concave (fast approx)", "value": "concave"},
                {"label": "Convex hull (fast, stable)", "value": "convex"},
            ],
            value="convex",
            style={"marginTop": "5px"}
        ),

        html.Div(id="envelope-help", style={"marginTop": "6px", "fontSize": "12px",
                                            "color": "#555", "lineHeight": "1.35"}),

        html.Button("Download interpolated CSV", id="btn-download", n_clicks=0,
                    style={"marginTop": "10px", "width": "100%"}),
        dcc.Download(id="download-data"),
    ], style={"flex": "0 0 340px", "overflowY": "auto", "height": "78vh",
              "paddingRight": "12px", "borderRight": "1px solid #eee"})

    envelope_graph = html.Div([
        dcc.Graph(id="graph", style={"height": "78vh"}),
        html.Div([
            df_to_dash_table(height_df, "Reference Matrix — Height [m]", "tbl-height"),
            html.Button("Download Height CSV", id="btn-height-csv", style={"margin": "8px 0 24px 0"}),
            dcc.Download(id="download-height"),
            df_to_dash_table(outreach_df, "Reference Matrix — Outreach [m]", "tbl-outreach"),
            html.Button("Download Outreach CSV", id="btn-outreach-csv", style={"margin": "8px 0 0 0"}),
            dcc.Download(id="download-outreach"),
        ], style={"marginTop": "16px"})
    ], style={"flex": "1 1 auto", "paddingLeft": "16px"})

    envelope_page = html.Div([envelope_controls, envelope_graph],
                             style={"display": "flex", "padding": "16px", "gap": "10px"})

    app.layout = html.Div([
        dcc.Location(id="url"),
        stores,
        header,
        html.Div([sidebar, html.Div(envelope_page, style={"flex": "1 1 auto"})],
                 style={"display": "flex", "gap": 0})
    ])

    @app.callback(Output("envelope-help", "children"), Input("envelope-type", "value"))
    def explain_envelope(kind):
        if kind == "none":
            return "Displays only points (no envelope). Useful for checking interpolation coverage."
        elif kind == "concave":
            return "Concave alpha shape: tighter, realistic shape but sensitive to sparse data."
        return "Convex hull: stable and conservative outer boundary."

    @app.callback(Output("graph", "figure"),
                  Input("main-factor", "value"),
                  Input("fold-factor", "value"),
                  Input("pedestal-toggle", "value"),
                  Input("envelope-type", "value"),
                  State("orig-xy", "data"),
                  State("orig-angles", "data"),
                  State("fold-angles", "data"),
                  State("main-angles", "data"))
    def update_figure(main_factor, fold_factor, pedestal_value, envelope_value,
                      orig_xy_store, orig_angles_store, fold_angles_store, main_angles_store):
        include_pedestal = "on" in (pedestal_value or [])
        main_factor, fold_factor = int(main_factor or 1), int(fold_factor or 1)
        f, m = np.array(fold_angles_store, float), np.array(main_angles_store, float)
        _, _, _, _, pts = resample_grid_by_factors
