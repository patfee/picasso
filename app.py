import os
import numpy as np
import pandas as pd

from flask import Flask
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from scipy.interpolate import RegularGridInterpolator
import shapely.geometry as sgeom
import alphashape

# ---------- Config ----------
DATA_DIR = os.getenv("DATA_DIR", "./data")
HEIGHT_CSV = os.getenv("HEIGHT_CSV", "height.csv")
OUTREACH_CSV = os.getenv("OUTREACH_CSV", "outreach.csv")

# ---------- Helpers ----------
def load_matrix_csv(path: str) -> pd.DataFrame:
    """
    Expects a matrix where:
      - Columns are Main-jib angles (deg)
      - Index/first column are Folding-jib angles (deg)
    Returns a DataFrame with float index and float columns.
    """
    df = pd.read_csv(path, index_col=0)
    # Coerce headers and index to float (strip degree symbols / spaces if any)
    df.columns = [float(str(c).replace("°", "").strip()) for c in df.columns]
    df.index = [float(str(i).replace("°", "").strip()) for i in df.index]
    df = df.sort_index().sort_index(axis=1)
    return df

def build_interpolators(height_df: pd.DataFrame, outreach_df: pd.DataFrame):
    """
    Returns RegularGridInterpolators over (fold_angle, main_angle).
    """
    fold_angles = height_df.index.values.astype(float)     # rows
    main_angles = height_df.columns.values.astype(float)   # cols

    H = height_df.values.astype(float)
    R = outreach_df.loc[fold_angles, main_angles].values.astype(float)

    # Linear is robust and respects monotonic circular motion assumption locally
    height_itp = RegularGridInterpolator((fold_angles, main_angles), H, bounds_error=False, fill_value=None)
    outre_itp  = RegularGridInterpolator((fold_angles, main_angles), R, bounds_error=False, fill_value=None)
    return (fold_angles, main_angles, height_itp, outre_itp)

def resample_grid(fold_angles, main_angles, d_fold: float, d_main: float):
    """
    Create dense angle grids with step in degrees chosen by the sliders.
    """
    fmin, fmax = float(np.min(fold_angles)), float(np.max(fold_angles))
    mmin, mmax = float(np.min(main_angles)), float(np.max(main_angles))

    # Ensure at least one step
    d_fold = max(0.1, float(d_fold))
    d_main = max(0.1, float(d_main))

    fnew = np.arange(fmin, fmax + 1e-6, d_fold)
    mnew = np.arange(mmin, mmax + 1e-6, d_main)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])
    return fnew, mnew, F, M, pts

def compute_boundary_curve(xy_points: np.ndarray):
    """
    Build a concave hull (alpha shape) to approximate the outer envelope,
    similar to the line in the reference image.
    """
    if len(xy_points) < 10:
        return None

    try:
        alpha = alphashape.optimizealpha(xy_points)
        poly = alphashape.alphashape(xy_points, alpha)
    except Exception:
        # Fall back to convex hull if alpha-shape fails
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

    # Dense/interpolated points
    fig.add_trace(go.Scatter(
        x=dense_xy[:, 0], y=dense_xy[:, 1],
        mode="markers",
        marker=dict(size=5, symbol="diamond"),
        name="Interpolated points"
    ))

    # Original grid points
    fig.add_trace(go.Scatter(
        x=orig_xy[:, 0], y=orig_xy[:, 1],
        mode="markers",
        marker=dict(size=8),
        name="Original matrix points"
    ))

    # Boundary line(s)
    if boundary:
        for bx, by in boundary:
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="lines",
                line=dict(width=3),
                name="Envelope"
            ))

    fig.update_layout(
        title="Tabular data points Main Hoist (Dash)",
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head above pedestal flange [m]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_yaxes(scaleanchor=None)  # free aspect like your image
    return fig

# ---------- App Factory ----------
def create_app():
    server = Flask(__name__)
    app = dash.Dash(
        __name__,
        server=server,
        suppress_callback_exceptions=True,
        title="Crane Curve Viewer (Dash)"
    )

    # Load once at startup
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))

    # The indices/columns are your angles:
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)

    # Store originals as (x = outreach, y = height) for quick plotting
    F0, M0 = np.meshgrid(fold_angles, main_angles, indexing="ij")
    orig_xy = np.column_stack([
        outreach_df.values.ravel(),
        height_df.values.ravel()
    ])

    app.layout = html.Div(
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "18px"},
        children=[
            html.H2("Crane Curve Viewer (Dash • Flask)"),
            html.P([
                "Columns = Main-jib angles (deg). Rows = Folding-jib angles (deg). ",
                "Interpolation assumes circular motion per jib (locally linear in angle space)."
            ]),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"}, children=[
                html.Div(children=[
                    html.Label("Step between Main-jib matrix angles (degrees)"),
                    dcc.Slider(
                        id="slider-main-step",
                        min=0.1, max=10, step=0.1, value=1.0,
                        marks=None, tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]),
                html.Div(children=[
                    html.Label("Step between Folding-jib matrix angles (degrees)"),
                    dcc.Slider(
                        id="slider-fold-step",
                        min=0.1, max=10, step=0.1, value=1.0,
                        marks=None, tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]),
            ]),
            html.Div(id="angle-summary", style={"marginTop": "8px", "fontFamily": "monospace"}),
            dcc.Graph(id="graph", style={"height": "78vh"}),
            # Hidden stores to avoid re-reading files
            dcc.Store(id="orig-xy", data=orig_xy.tolist()),
            dcc.Store(id="fold-angles", data=fold_angles.tolist()),
            dcc.Store(id="main-angles", data=main_angles.tolist()),
        ]
    )

    @app.callback(
        Output("graph", "figure"),
        Output("angle-summary", "children"),
        Input("slider-main-step", "value"),
        Input("slider-fold-step", "value"),
        State("orig-xy", "data"),
        State("fold-angles", "data"),
        State("main-angles", "data"),
        prevent_initial_call=False
    )
    def update_figure(main_step_deg, fold_step_deg, orig_xy_store, fold_angles_store, main_angles_store):
        # Rebuild dense angle grid
        fnew, mnew, F, M, pts = resample_grid(
            np.array(fold_angles_store), np.array(main_angles_store),
            d_fold=fold_step_deg, d_main=main_step_deg
        )

        # Interpolate using the regular grid interpolators
        # NOTE: these closures capture height_itp & outre_itp from above
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)

        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]  # drop edge NaNs

        # Compute concave boundary (envelope)
        boundary = compute_boundary_curve(dense_xy)

        fig = make_figure(np.array(orig_xy_store), dense_xy, boundary)
        summary = f"Interpolated grid → Main step: {main_step_deg:.2f}° | Folding step: {fold_step_deg:.2f}° | Points: {len(dense_xy)}"
        return fig, summary

    return app, server

# For gunicorn
app, server = create_app()

if __name__ == "__main__":
    # Handy for local testing
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)
