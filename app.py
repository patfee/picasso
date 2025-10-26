import os
import csv
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
        head = f.read(4096)
    delim = _sniff_delimiter(head)

    df = pd.read_csv(path, sep=delim, engine="python")

    # Assume first column is the row labels (folding-jib angles)
    first_col = df.columns[0]
    # If first column looks like angle labels or is a common label name, use it as index
    if (
        first_col.lower().strip() in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
        or df[first_col].dropna().astype(str).str.replace(",", ".", regex=False)
           .str.match(r"^\s*-?\d+(\.\d+)?\s*$").all()
    ):
        df = df.set_index(first_col)

    # Clean headers and index to floats
    df.columns = _clean_angles(df.columns)
    df.index = _clean_angles(df.index)

    # Drop non-numeric/empty
    df = df.loc[~df.index.isna(), :]
    df = df.loc[:, ~df.columns.isna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Sort angles ascending
    df = df.sort_index().sort_index(axis=1)

    print(f"[load_matrix_csv] {path}: shape={df.shape}, delim='{delim}'")
    if not df.empty:
        print(
            f"[load_matrix_csv] fold_angles={df.index.min()}..{df.index.max()} | "
            f"main_angles={df.columns.min()}..{df.columns.max()}"
        )

    if df.empty:
        raise ValueError(f"{path} parsed to an empty DataFrame. Check delimiter and layout.")
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

    # Align outreach to same axes just in case
    outreach_df = outreach_df.loc[fold_angles, main_angles]

    H = height_df.values.astype(float)
    R = outreach_df.values.astype(float)

    # Linear interpolation across angle space (angle parameterization matches your perfect-circle assumption per jib)
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

    boundary_traces = []
    if isinstance(poly, sgeom.Polygon):
        x, y = poly.exterior.coords.xy
        boundary_traces.append((np.asarray(x), np.asarray(y)))
    elif isinstance(poly, sgeom.MultiPolygon):
        for p in poly.geoms:
            x, y = p.exterior.coords.xy
            boundary_traces.append((np.asarray(x), np.asarray(y)))
    else:
        return None
    return boundary_traces


def make_figure(orig_xy: np.ndarray, dense_xy: np.ndarray, boundary):
    """
    Produce a Plotly figure: dense grid (small diamonds), original grid (bigger dots),
    and an outer boundary line.
    """
    fig = go.Figure()

    # Interpolated points
    if dense_xy.size:
        fig.add_trace(
            go.Scatter(
                x=dense_xy[:, 0],
                y=dense_xy[:, 1],
                mode="markers",
                marker=dict(size=5, symbol="diamond"),
                name="Interpolated points",
            )
        )

    # Original grid points
    if orig_xy.size:
        fig.add_trace(
            go.Scatter(
                x=orig_xy[:, 0],
                y=orig_xy[:, 1],
                mode="markers",
                marker=dict(size=8),
                name="Original matrix points",
            )
        )

    # Envelope
    if boundary:
        for bx, by in boundary:
            fig.add_trace(
                go.Scatter(x=bx, y=by, mode="lines", line=dict(width=3), name="Envelope")
            )

    fig.update_layout(
        title="Tabular data points Main Hoist (Dash)",
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head above pedestal flange [m]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    # Free aspect like the reference image (no fixed scale anchor)
    fig.update_yaxes(scaleanchor=None)
    return fig


# =========================
# App Factory
# =========================
def create_app():
    server = Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        suppress_callback_exceptions=True,
        title="Crane Curve Viewer (Dash)",
    )

    # Load CSV matrices once at startup
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))

    # Interpolators over (fold_angle, main_angle)
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)

    # Original points (x = outreach, y = height)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    app.layout = html.Div(
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "18px"},
        children=[
            html.H2("Crane Curve Viewer (Dash • Flask)"),
            html.P(
                [
                    "Columns = Main-jib angles (deg). Rows = Folding-jib angles (deg). ",
                    "Interpolation assumes circular motion per jib (angle-parameterized). ",
                    "Main 0° = horizontal → increasing = up; Folding 0° = flat against main → increasing = out.",
                ]
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"},
                children=[
                    html.Div(
                        children=[
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
                        ]
                    ),
                    html.Div(
                        children=[
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
                        ]
                    ),
                ],
            ),
            html.Div(id="angle-summary", style={"marginTop": "8px", "fontFamily": "monospace"}),
            dcc.Graph(id="graph", style={"height": "78vh"}),
            # Hidden stores to avoid re-reading files
            dcc.Store(id="orig-xy", data=orig_xy.tolist()),
            dcc.Store(id="fold-angles", data=fold_angles.tolist()),
            dcc.Store(id="main-angles", data=main_angles.tolist()),
        ],
    )

    @app.callback(
        Output("graph", "figure"),
        Output("angle-summary", "children"),
        Input("slider-main-step", "value"),
        Input("slider-fold-step", "value"),
        State("orig-xy", "data"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
        prevent_initial_call=False,
    )
    def update_figure(main_step_deg, fold_step_deg, orig_xy_store, fold_angles_store, main_angles_store):
        # Dense angle grid
        fnew, mnew, F, M, pts = resample_grid(
            np.array(fold_angles_store), np.array(main_angles_store), d_fold=fold_step_deg, d_main=main_step_deg
        )

        # Interpolate Height & Outreach on dense grid
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)

        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

        # Envelope (concave hull)
        boundary = compute_boundary_curve(dense_xy)

        fig = make_figure(np.array(orig_xy_store), dense_xy, boundary)
        summary = (
            f"Interpolated grid → Main step: {float(main_step_deg):.2f}° | "
            f"Folding step: {float(fold_step_deg):.2f}° | Points: {len(dense_xy)}"
        )
        return fig, summary

    return app, server


# ================
# Entry points
# ================
app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)
