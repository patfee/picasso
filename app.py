import os
import csv
import io
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
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
            first_col.lower().strip() in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
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

    height_itp = RegularGridInterpolator((fold_angles, main_angles), height_df.values,
                                         bounds_error=False, fill_value=None)
    outre_itp  = RegularGridInterpolator((fold_angles, main_angles), outreach_df.values,
                                         bounds_error=False, fill_value=None)
    return fold_angles, main_angles, height_itp, outre_itp

def resample_grid(fold_angles, main_angles, d_fold: float, d_main: float):
    fmin, fmax = float(np.min(fold_angles)), float(np.max(fold_angles))
    mmin, mmax = float(np.min(main_angles)), float(np.max(main_angles))
    d_fold, d_main = max(0.5, d_fold), max(0.5, d_main)
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
    )
    return fig

def _three_marks(vmin: float, vmax: float, unit: str):
    mid = (vmin + vmax) / 2.0
    style = {"fontSize": "11px", "color": "#555"}
    return {
        round(vmin, 2): {"label": f"{vmin:g}{unit}", "style": style},
        round(mid,  2): {"label": f"{mid:g}{unit}",  "style": style},
        round(vmax, 2): {"label": f"{vmax:g}{unit}", "style": style},
    }

def df_to_dash_table(df: pd.DataFrame, title: str, table_id: str):
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

    # Data
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    # Stores
    stores = html.Div([
        dcc.Store(id="orig-xy", data=orig_xy.tolist()),
        dcc.Store(id="fold-angles", data=fold_angles.tolist()),
        dcc.Store(id="main-angles", data=main_angles.tolist()),
    ])

    # Sidebar
    sidebar = html.Div([
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
        html.Br(),
        dcc.Link("Test", href="/test", className="nav-link"),
    ], style={
        "position": "sticky", "top": "0px", "height": "100vh",
        "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa",
    })

    # Header
    header = html.Div([
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(
            src=LOGO_URL,
            style={"height": "42px", "objectFit": "contain", "position": "absolute",
                   "right": "16px", "top": "12px"},
            alt="DCN Diving logo",
        ),
    ], style={"position": "relative", "padding": "12px 16px 4px 16px"})

    # ====== Controls (compact sliders + inputs) ======
    def slider_row(slider_id, input_id):
        return html.Div([
            dcc.Slider(
                id=slider_id, min=0.5, max=5.0, step=0.1, value=1.0,
                included=False, updatemode="mouseup",
                marks=_three_marks(0.5, 5.0, "°"),
                tooltip={"placement": "bottom", "always_visible": False},
                style={"flex": "1 1 auto", "marginRight": "10px", "height": "24px"}
            ),
            dcc.Input(
                id=input_id, type="number", value=1.0,
                min=0.5, max=5.0, step=0.1,
                style={"width": "90px", "height": "28px", "fontFamily": "monospace"}
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "6px"})

    envelope_controls = html.Div([
        html.Label("Step between Main-jib matrix angles (°)", style={"fontWeight": 600}),
        slider_row("slider-main-step", "input-main-step"),
        html.Div(id="main-range", style={"marginTop": "2px", "fontFamily": "monospace", "color": "#444"}),

        html.Label("Step between Folding-jib matrix angles (°)", style={"fontWeight": 600, "marginTop": "10px"}),
        slider_row("slider-fold-step", "input-fold-step"),
        html.Div(id="fold-range", style={"marginTop": "2px", "fontFamily": "monospace", "color": "#444"}),

        dcc.Checklist(
            id="pedestal-toggle",
            options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
            value=["on"], style={"marginTop": "12px"}
        ),

        html.Label("Envelope type", style={"fontWeight": 600, "marginTop": "6px"}),
        dcc.RadioItems(
            id="envelope-type",
            options=[{"label": "Concave (fast approx)", "value": "concave"},
                     {"label": "Convex hull (fast, stable)", "value": "convex"}],
            value="convex",
            style={"marginTop": "5px"}
        ),

        html.Div(id="envelope-help",
                 style={"marginTop": "6px", "fontSize": "12px", "color": "#555", "lineHeight": "1.35"}),

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

    # Graph + tables area
    envelope_graph = html.Div(
        [
            dcc.Graph(id="graph", style={"height": "78vh"}),
            html.Div([
                df_to_dash_table(height_df, "Reference Matrix — Height [m]", "tbl-height"),
                html.Button("Download Height CSV", id="btn-height-csv", style={"margin": "8px 0 24px 0"}),
                dcc.Download(id="download-height"),

                df_to_dash_table(outreach_df, "Reference Matrix — Outreach [m]", "tbl-outreach"),
                html.Button("Download Outreach CSV", id="btn-outreach-csv", style={"margin": "8px 0 0 0"}),
                dcc.Download(id="download-outreach"),
            ], style={"marginTop": "16px"})
        ],
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

    # Routing
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def route(pathname: str):
        if pathname in ("/", "/envelope", None):
            return envelope_page
        elif pathname == "/test":
            return test_page
        return html.Div([html.H3("404"), html.P(f"Path not found: {pathname}")], style={"padding": "16px"})

    # Ranges text
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

    # Explain envelope type
    @app.callback(Output("envelope-help", "children"), Input("envelope-type", "value"))
    def explain_envelope(kind):
        if kind == "concave":
            return html.Div([
                html.B("Concave (alpha shape): "),
                html.Span("Tightly wraps the points and follows inward bends. Closer to 'true' envelope; more sensitive.")
            ])
        return html.Div([
            html.B("Convex hull: "),
            html.Span("Smallest convex polygon enclosing all points. Ignores concavities; fast and stable.")
        ])

    # --- Dynamic bounds for step controls (min=0.5, max=angle span) ---
    @app.callback(
        Output("slider-main-step", "min"),
        Output("slider-main-step", "max"),
        Output("slider-main-step", "marks"),
        Output("input-main-step", "min"),
        Output("input-main-step", "max"),
        Output("slider-fold-step", "min"),
        Output("slider-fold-step", "max"),
        Output("slider-fold-step", "marks"),
        Output("input-fold-step", "min"),
        Output("input-fold-step", "max"),
        Input("main-angles", "data"),
        Input("fold-angles", "data"),
    )
    def update_step_bounds(main_angles_store, fold_angles_store):
        m = np.array(main_angles_store, float)
        f = np.array(fold_angles_store, float)
        m_span = max(0.5, float(m.max() - m.min()))
        f_span = max(0.5, float(f.max() - f.min()))
        return (0.5, m_span, _three_marks(0.5, m_span, "°"), 0.5, m_span,
                0.5, f_span, _three_marks(0.5, f_span, "°"), 0.5, f_span)

    # --- Keep slider and numeric input linked (Main) ---
    @app.callback(
        Output("slider-main-step", "value"),
        Output("input-main-step", "value"),
        Input("slider-main-step", "value"),
        Input("input-main-step", "value"),
        State("slider-main-step", "min"),
        State("slider-main-step", "max"),
        prevent_initial_call=True,
    )
    def sync_main(slider_val, input_val, vmin, vmax):
        ctx_id = dash.ctx.triggered_id
        def clamp(v): 
            try: 
                return float(min(max(v, vmin), vmax))
            except Exception:
                return vmin
        if ctx_id == "slider-main-step":
            v = clamp(slider_val)
        elif ctx_id == "input-main-step":
            v = clamp(input_val)
        else:
            return no_update, no_update
        return v, v

    # --- Keep slider and numeric input linked (Folding) ---
    @app.callback(
        Output("slider-fold-step", "value"),
        Output("input-fold-step", "value"),
        Input("slider-fold-step", "value"),
        Input("input-fold-step", "value"),
        State("slider-fold-step", "min"),
        State("slider-fold-step", "max"),
        prevent_initial_call=True,
    )
    def sync_fold(slider_val, input_val, vmin, vmax):
        ctx_id = dash.ctx.triggered_id
        def clamp(v):
            try:
                return float(min(max(v, vmin), vmax))
            except Exception:
                return vmin
        if ctx_id == "slider-fold-step":
            v = clamp(slider_val)
        elif ctx_id == "input-fold-step":
            v = clamp(input_val)
        else:
            return no_update, no_update
        return v, v

    # Graph update (use the numeric inputs; sliders are synced anyway)
    @app.callback(
        Output("graph", "figure"),
        Input("input-main-step", "value"),
        Input("input-fold-step", "value"),
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
                                        d_fold=float(fold_step or 1.0), d_main=float(main_step or 1.0))
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

    # Download interpolated grid
    @app.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        State("input-main-step", "value"),
        State("input-fold-step", "value"),
        State("pedestal-toggle", "value"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
        prevent_initial_call=True,
    )
    def download_interpolated(n_clicks, main_step, fold_step, pedestal_value, fold_angles_store, main_angles_store):
        if not n_clicks:
            return no_update
        include_pedestal = "on" in (pedestal_value or [])
        _, _, _, _, pts = resample_grid(np.array(fold_angles_store), np.array(main_angles_store),
                                        d_fold=float(fold_step or 1.0), d_main=float(main_step or 1.0))
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M
        df_out = pd.DataFrame({
            "FoldingJib_deg": pts[:, 0],
            "MainJib_deg":    pts[:, 1],
            "Outreach_m":     R_dense,
            "Height_m":       H_dense,
            "PedestalIncluded": include_pedestal,
        }).dropna()
        return dcc.send_data_frame(df_out.to_csv, "interpolated_envelope.csv", index=False)

    # Download original matrices
    @app.callback(Output("download-height", "data"), Input("btn-height-csv", "n_clicks"), prevent_initial_call=True)
    def download_height_matrix(n):
        return dcc.send_data_frame(height_df.to_csv, "height_matrix.csv")

    @app.callback(Output("download-outreach", "data"), Input("btn-outreach-csv", "n_clicks"), prevent_initial_call=True)
    def download_outreach_matrix(n):
        return dcc.send_data_frame(outreach_df.to_csv, "outreach_matrix.csv")

    return app, server

# =========================
# Entrypoint
# =========================
app, server = create_app()

if __name__ == "__main__":
    # Production: gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
