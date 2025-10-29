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
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")

MAX_POINTS_FOR_DISPLAY = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN = 1200
KNN_K = 8


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
            return np.column_stack([np.asarray(x), np.asarray(y)])
        if isinstance(poly, sgeom.LineString):
            x, y = poly.coords.xy
            return np.column_stack([np.asarray(x), np.asarray(y)])
    except Exception:
        pass
    return None


# =========================
# Plot utilities
# =========================
def make_heatmap(df: pd.DataFrame, title: str, colorscale="Viridis"):
    """Render a matrix as a heatmap for visualizing input data."""
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale=colorscale,
            colorbar=dict(title=title),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Main-jib angle [°]",
        yaxis_title="Folding-jib angle [°]",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
    )
    return fig


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
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head height [m]" if include_pedestal else "Jib head above pedestal flange [m]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="keep",
        height=600
    )
    return fig


def _three_marks(vmin: float, vmax: float, unit: str):
    mid = (vmin + vmax) / 2.0
    style = {"fontSize": "12px", "color": "#555"}
    return {
        round(vmin, 2): {"label": f"{vmin:g}{unit}", "style": style},
        round(mid,  2): {"label": f"{mid:g}{unit}",  "style": style},
        round(vmax, 2): {"label": f"{vmax:g}{unit}", "style": style},
    }


# =========================
# App Factory (multi-page)
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True,
                    title="DCN Picasso DSV Engineering Data")

    # Load data
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    step_min, step_max = 0.1, 10.0
    step_marks = _three_marks(step_min, step_max, "°")

    stores = html.Div([
        dcc.Store(id="orig-xy", data=orig_xy.tolist()),
        dcc.Store(id="fold-angles", data=fold_angles.tolist()),
        dcc.Store(id="main-angles", data=main_angles.tolist()),
    ])

    sidebar = html.Div([
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
        html.Br(),
        dcc.Link("Test", href="/test", className="nav-link"),
    ], style={"position": "sticky", "top": "0px", "height": "100vh",
              "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa"})

    header = html.Div([
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(src=LOGO_URL,
                 style={"height": "42px", "objectFit": "contain",
                        "position": "absolute", "right": "16px", "top": "12px"},
                 alt="DCN Diving logo"),
    ], style={"position": "relative", "padding": "12px 16px 4px 16px"})

    envelope_controls = html.Div([
        html.Label("Step between Main-jib matrix angles (°)", style={"fontWeight": 600}),
        dcc.Slider(id="slider-main-step", min=step_min, max=step_max, step=0.1, value=1.0,
                   marks=step_marks, included=False, updatemode="mouseup",
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(id="main-range", style={"marginTop": "6px", "fontFamily": "monospace", "color": "#444"}),

        html.Label("Step between Folding-jib matrix angles (°)", style={"fontWeight": 600, "marginTop": "10px"}),
        dcc.Slider(id="slider-fold-step", min=step_min, max=step_max, step=0.1, value=1.0,
                   marks=step_marks, included=False, updatemode="mouseup",
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Div(id="fold-range", style={"marginTop": "6px", "fontFamily": "monospace", "color": "#444"}),

        dcc.Checklist(id="pedestal-toggle",
                      options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
                      value=["on"], style={"marginTop": "12px"}),

        html.Label("Envelope type", style={"fontWeight": 600, "marginTop": "6px"}),
        dcc.RadioItems(
            id="envelope-type",
            options=[{"label": "Concave (fast approx)", "value": "concave"},
                     {"label": "Convex hull (fast, stable)", "value": "convex"}],
            value="convex",
            style={"marginTop": "5px"}
        ),

        html.Div(
            id="envelope-help",
            style={"marginTop": "6px", "fontSize": "12px", "color": "#555", "lineHeight": "1.35"}
        ),

        html.Div([
            html.Button("Download interpolated CSV", id="btn-download", n_clicks=0,
                        style={"marginTop": "10px", "width": "100%"}),
            dcc.Download(id="download-data"),
        ], style={"position": "sticky", "bottom": 0, "background": "#fafafa", "padding": "8px 0"}),

    ], style={"flex": "0 0 340px", "overflowY": "auto", "height": "78vh",
              "paddingRight": "12px", "borderRight": "1px solid #eee"})

    # Graph + Matrices
    envelope_graph = html.Div([
        dcc.Graph(id="graph", style={"height": "78vh"}),
        html.Div([
            html.H4("Original Matrices", style={"marginTop": "12px"}),
            html.Div([
                html.Div([dcc.Graph(id="height-matrix", figure=make_heatmap(height_df, "Height [m]"))],
                         style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div([dcc.Graph(id="outreach-matrix", figure=make_heatmap(outreach_df, "Outreach [m]"))],
                         style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
            ]),
        ], style={"marginTop": "20px"})
    ], style={"flex": "1 1 auto", "paddingLeft": "16px"})

    envelope_page = html.Div(
        [envelope_controls, envelope_graph],
        style={"display": "flex", "padding": "16px", "gap": "10px"}
    )

    test_page = html.Div([html.H3("Test Page"), html.P("Placeholder for future content.")],
                         style={"padding": "16px"})
    content = html.Div(id="page-content", style={"width": "100%"})

    app.layout = html.Div([
        dcc.Location(id="url"), stores, header,
        html.Div([
            html.Div(sidebar, style={"width": "260px", "flex": "0 0 260px"}),
            html.Div(content, style={"flex": "1 1 auto"}),
        ], style={"display": "flex", "gap": "0px"}),
    ])

    # Routing
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def route(pathname: str):
        if pathname in ("/", "/envelope", None):
            return envelope_page
        elif pathname == "/test":
            return test_page
        return html.Div([html.H3("404"), html.P(f"Path not found: {pathname}")], style={"padding": "16px"})

    # Envelope text helper
    @app.callback(Output("envelope-help", "children"), Input("envelope-type", "value"))
    def explain_envelope(kind):
        if kind == "concave":
            return html.Div([
                html.B("Concave (alpha shape): "),
                html.Span("Follows the inner curves of the dataset. More accurate but sensitive to noise or sparse points.")
            ])
        return html.Div([
            html.B("Convex hull: "),
            html.Span("Encloses all points in a smooth convex shape. Fast and robust, but can overestimate the area.")
        ])

    return app, server


# Entrypoint
app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 300
