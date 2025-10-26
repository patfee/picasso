import os
import csv
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State, callback_context
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
PEDESTAL_HEIGHT_M = float(os.getenv("PEDESTAL_HEIGHT_M", "6"))  # 6 m
LOGO_URL = os.getenv(
    "LOGO_URL",
    "https://www.google.com/url?sa=i&url=https%3A%2F%2Fdcndiving.com%2F&psig=AOvVaw08weDg3_w7rLPAiYloISj5&ust=1761566373095000&source=images&cd=vfe&opi=89978449&ved=0CBUQjRxqFwoTCPj8qaPowZADFQAAAAAdAAAAABAE",
)  # Replace with a direct PNG/SVG if you have one


# =========================
# CSV Loading (robust to ; , \t and messy headers)
# =========================
def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        return ";"  # EU-style CSV fallback


def _clean_angles(vals):
    cleaned = (
        pd.Series(vals)
        .astype(str)
        .str.replace("°", "", regex=False)
        .str.replace("deg", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)  # allow decimal comma
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_matrix_csv(path: str) -> pd.DataFrame:
    """
    Loads a jib matrix where:
      - Columns = Main-jib angles (deg)
      - Rows (index) = Folding-jib angles (deg)
    Handles delimiters (, ; \t), degree symbols, decimal commas, and common first-column labels.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Delimiter detection
    delim = _sniff_delimiter(raw[:4096])

    # Try pandas auto-detect first
    try:
        df = pd.read_csv(pd.compat.StringIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        df = pd.read_csv(pd.compat.StringIO(raw), sep=delim, engine="python")
        first_col = df.columns[0]
        if (
            first_col.lower().strip() in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
            or df[first_col].dropna().astype(str).str.replace(",", ".", regex=False)
            .str.match(r"^\s*-?\d+(\.\d+)?\s*$").all()
        ):
            df = df.set_index(first_col)

    # Clean headers and index to floats
    df.columns = _clean_angles(df.columns)
    df.index = _clean_angles(df.index)

    # Coerce values to numeric (supports decimal comma inside cells)
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))

    # Drop fully-empty rows/cols and sort axes
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.sort_index().sort_index(axis=1)

    print(f"[load_matrix_csv] {path}: shape={df.shape}, delim='{delim}'")
    if not df.empty:
        print(
            f"[load_matrix_csv] fold_angles={df.index.min()}..{df.index.max()} | "
            f"main_angles={df.columns.min()}..{df.columns.max()}"
        )

    if df.empty:
        raise ValueError(f"{path} parsed to an empty DataFrame. Check delimiter/layout.")
    return df


# =========================
# Interpolation & Geometry
# =========================
def build_interpolators(height_df: pd.DataFrame, outreach_df: pd.DataFrame):
    """
    Returns RegularGridInterpolators over (fold_angle, main_angle).
    Assumes both dataframes share same index/columns.
    """
    fold_angles = height_df.index.values.astype(float)     # rows
    main_angles = height_df.columns.values.astype(float)   # cols

    outreach_df = outreach_df.loc[fold_angles, main_angles]

    H = height_df.values.astype(float)
    R = outreach_df.values.astype(float)

    height_itp = RegularGridInterpolator(
        (fold_angles, main_angles), H, bounds_error=False, fill_value=None
    )
    outre_itp = RegularGridInterpolator(
        (fold_angles, main_angles), R, bounds_error=False, fill_value=None
    )
    return (fold_angles, main_angles, height_itp, outre_itp)


def resample_grid(fold_angles, main_angles, d_fold: float, d_main: float):
    """
    Create dense angle grids with step in degrees chosen by the sliders.
    """
    fmin, fmax = float(np.min(fold_angles)), float(np.max(fold_angles))
    mmin, mmax = float(np.min(main_angles)), float(np.max(main_angles))

    d_fold = max(0.1, float(d_fold))
    d_main = max(0.1, float(d_main))

    fnew = np.arange(fmin, fmax + 1e-9, d_fold)
    mnew = np.arange(mmin, mmax + 1e-9, d_main)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])
    return fnew, mnew, F, M, pts


def compute_boundary_curve(xy_points: np.ndarray):
    """
    Build a concave hull (alpha shape) to approximate the outer envelope.
    Fallback to convex hull if alpha-shape fails.
    """
    if len(xy_points) < 10:
        return None

    try:
        alpha = alphashape.optimizealpha(xy_points)
        poly = alphashape.alphashape(xy_points, alpha)
    except Exception:
        poly = sgeom.MultiPoint(xy_points).convex_hull

    if isinstance(poly, sgeom.Polygon):
        x, y = poly.exterior.coords.xy
        return np.column_stack([np.array(x), np.array(y)])
    elif isinstance(poly, sgeom.MultiPolygon):
        # take the largest polygon
        p = max(poly.geoms, key=lambda g: g.area)
        x, y = p.exterior.coords.xy
        return np.column_stack([np.array(x), np.array(y)])
    return None


def make_figure(orig_xy: np.ndarray, dense_xy: np.ndarray, boundary_xy, include_pedestal: bool):
    """
    Produce a Plotly figure: dense grid, original grid, and an outer boundary line.
    """
    fig = go.Figure()

    # Interpolated points
    if dense_xy.size:
        fig.add_trace(
            go.Scatter(
                x=dense_xy[:, 0],
                y=dense_xy[:, 1],
                mode="markers",
                marker=dict(size=4, symbol="diamond", opacity=0.5),
                name="Interpolated points",
            )
        )

    # Original grid points
    if orig_xy.size:
        fig.add_trace(
            go.Scatter(
                x=orig_xy[:, 0],
                y=orig_xy[:, 1] + (PEDESTAL_HEIGHT_M if include_pedestal else 0.0),
                mode="markers",
                marker=dict(size=8),
                name="Original matrix points",
            )
        )

    # Envelope
    if boundary_xy is not None and len(boundary_xy) > 2:
        fig.add_trace(
            go.Scatter(
                x=boundary_xy[:, 0],
                y=boundary_xy[:, 1],
                mode="lines",
                line=dict(width=3),
                name="Envelope",
            )
        )

    y_label = "Jib head height incl. pedestal [m]" if include_pedestal else "Jib head above pedestal flange [m]"
    fig.update_layout(
        title="DCN Picasso DSV Engineering Data",
        xaxis_title="Outreach [m]",
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# =========================
# App Factory (Multi-page)
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        suppress_callback_exceptions=True,
        title="DCN Picasso DSV Engineering Data",
    )

    # Load CSV matrices once at startup
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))

    # Interpolators over (fold_angle, main_angle)
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)

    # Original points (x = outreach, y = height above pedestal)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    # ------------- Shared Stores -------------
    stores = html.Div(
        [
            dcc.Store(id="orig-xy", data=orig_xy.tolist()),
            dcc.Store(id="fold-angles", data=fold_angles.tolist()),
            dcc.Store(id="main-angles", data=main_angles.tolist()),
        ]
    )

    # ------------- Sidebar (navigation) -------------
    sidebar = html.Div(
        [
            html.H3("Menu", style={"marginTop": 0}),
            dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
            html.Br(),
            dcc.Link("Test", href="/test", className="nav-link"),
        ],
        style={
            "position": "sticky",
            "top": "0px",
            "height": "100vh",
            "padding": "16px",
            "borderRight": "1px solid #e5e7eb",
            "background": "#fafafa",
        },
    )

    # ------------- Header (title + logo) -------------
    header = html.Div(
        [
            html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
            html.Img(
                src=LOGO_URL,
                style={
                    "height": "42px",
                    "objectFit": "contain",
                    "position": "absolute",
                    "right": "16px",
                    "top": "12px",
                },
                alt="DCN Diving logo",
            ),
        ],
        style={"position": "relative", "padding": "12px 16px 4px 16px"},
    )

    # ------------- Envelope Page Content -------------
    envelope_controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Step between Main-jib matrix angles (degrees)"),
                    dcc.Slider(
                        id="slider-main-step",
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1.0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Div(id="main-range", style={"marginTop": "6px", "fontFamily": "monospace"}),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                [
                    html.Label("Step between Folding-jib matrix angles (degrees)"),
                    dcc.Slider(
                        id="slider-fold-step",
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1.0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Div(id="fold-range", style={"marginTop": "6px", "fontFamily": "monospace"}),
                ],
                style={"marginBottom": "16px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="pedestal-toggle",
                        options=[{"label": f"Include pedestal height (+{PEDESTAL_HEIGHT_M:.1f} m)", "value": "on"}],
                        value=["on"],  # default ON
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "inline-block", "marginRight": "12px"},
                        style={"marginTop": "4px"},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Button("Download interpolated CSV", id="btn-download", n_clicks=0),
                    dcc.Download(id="download-data"),
                ],
                style={"marginBottom": "12px"},
            ),
        ]
    )

    envelope_graph = dcc.Graph(id="graph", style={"height": "78vh"})

    envelope_page = html.Div(
        [
            envelope_controls,
            envelope_graph,
        ],
        style={"padding": "16px"},
    )

    # ------------- Test Page (blank placeholder) -------------
    test_page = html.Div(
        [
            html.H3("Test Page"),
            html.P("This is a placeholder page for future content."),
        ],
        style={"padding": "16px"},
    )

    # ------------- Router -------------
    content = html.Div(id="page-content", style={"width": "100%"})

    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            stores,
            header,
            html.Div(
                [
                    html.Div(sidebar, style={"width": "260px", "flex": "0 0 260px"}),
                    html.Div(content, style={"flex": "1 1 auto"}),
                ],
                style={"display": "flex", "gap": "0px"},
            ),
        ]
    )

    # ============== Callbacks (Routing) ==============
    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def route(pathname: str):
        if pathname in ("/", "/envelope", None):
            return envelope_page
        elif pathname == "/test":
            return test_page
        return html.Div([html.H3("404"), html.P(f"Path not found: {pathname}")], style={"padding": "16px"})

    # ============== Callbacks (UI Info) ==============
    @app.callback(
        Output("main-range", "children"),
        Output("fold-range", "children"),
        Input("main-angles", "data"),
        Input("fold-angles", "data"),
        prevent_initial_call=False,
    )
    def show_ranges(main_angles_store, fold_angles_store):
        main_arr = np.array(main_angles_store, dtype=float)
        fold_arr = np.array(fold_angles_store, dtype=float)
        main_txt = f"Main-jib angle range: {main_arr.min():.2f}° – {main_arr.max():.2f}°"
        fold_txt = f"Folding-jib angle range: {fold_arr.min():.2f}° – {fold_arr.max():.2f}°"
        return main_txt, fold_txt

    # ============== Callbacks (Graph) ==============
    @app.callback(
        Output("graph", "figure"),
        Input("slider-main-step", "value"),
        Input("slider-fold-step", "value"),
        Input("pedestal-toggle", "value"),
        State("orig-xy", "data"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
        prevent_initial_call=False,
    )
    def update_figure(main_step_deg, fold_step_deg, pedestal_value, orig_xy_store, fold_angles_store, main_angles_store):
        include_pedestal = "on" in (pedestal_value or [])

        # Dense angle grid
        fnew, mnew, F, M, pts = resample_grid(
            np.array(fold_angles_store), np.array(main_angles_store), d_fold=fold_step_deg, d_main=main_step_deg
        )

        # Interpolate Height & Outreach on dense grid
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)

        # Adjust height by pedestal if enabled
        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M

        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

        # Envelope (concave hull)
        boundary_xy = compute_boundary_curve(dense_xy)

        fig = make_figure(np.array(orig_xy_store), np.array(orig_xy_store) if dense_xy.size == 0 else np.column_stack([R_dense, H_dense]), boundary_xy, include_pedestal)
        return fig

    # ============== Callbacks (Download) ==============
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
    def download_interpolated(n_clicks, main_step_deg, fold_step_deg, pedestal_value, fold_angles_store, main_angles_store):
        if not n_clicks:
            return dash.no_update

        include_pedestal = "on" in (pedestal_value or [])

        # Build grid at current resolution
        _, _, _, _, pts = resample_grid(
            np.array(fold_angles_store), np.array(main_angles_store), d_fold=fold_step_deg, d_main=main_step_deg
        )

        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)

        if include_pedestal:
            H_dense = H_dense + PEDESTAL_HEIGHT_M

        # Build a DataFrame for export
        df_out = pd.DataFrame(
            {
                "FoldingJib_deg": pts[:, 0],
                "MainJib_deg": pts[:, 1],
                "Outreach_m": R_dense,
                "Height_m": H_dense,
                "PedestalIncluded": include_pedestal,
            }
        ).dropna()

        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        return dcc.send_bytes(lambda b: b.write(csv_bytes), filename="interpolated_envelope.csv")

    return app, server


# ================
# Entry points
# ================
app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)
