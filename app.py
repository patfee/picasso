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

# ============================================================
# Utility helpers
# ============================================================

def _sniff_delimiter(filepath):
    """Auto-detect delimiter (; , or tab)."""
    with open(filepath, "r", encoding="utf-8") as f:
        head = f.readline()
    for d in [";", ",", "\t"]:
        if d in head:
            return d
    return ","


def _clean_angles(series):
    """Convert angles like '10°' or '10,5' to float degrees."""
    cleaned = (
        series.astype(str)
        .str.replace("°", "", regex=False)
        .str.replace("deg", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_matrix_csv(filepath):
    """
    Load a matrix-style CSV: rows = Folding jib, cols = Main jib.
    First row = main angles, first column = folding angles.
    """
    delim = _sniff_delimiter(filepath)
    df = pd.read_csv(filepath, sep=delim, header=0, index_col=0)
    df.index = _clean_angles(df.index)
    df.columns = _clean_angles(df.columns)
    df = df.apply(pd.to_numeric, errors="coerce")
    print(f"[load_matrix_csv] {os.path.basename(filepath)}: shape={df.shape}, delim='{delim}'")
    return df


def build_interpolators(height_df, outreach_df):
    """Build RegularGridInterpolators for height and outreach matrices."""
    fold_angles = height_df.index.values
    main_angles = height_df.columns.values
    H = height_df.values
    R = outreach_df.values
    H_itp = RegularGridInterpolator((fold_angles, main_angles), H, bounds_error=False, fill_value=np.nan)
    R_itp = RegularGridInterpolator((fold_angles, main_angles), R, bounds_error=False, fill_value=np.nan)
    return fold_angles, main_angles, H_itp, R_itp


def resample_grid(fold_angles, main_angles, d_fold=1.0, d_main=1.0):
    """Generate a regular grid of new angles and query points for interpolation."""
    f_new = np.arange(fold_angles.min(), fold_angles.max() + d_fold, d_fold)
    m_new = np.arange(main_angles.min(), main_angles.max() + d_main, d_main)
    F, M = np.meshgrid(f_new, m_new, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])
    return f_new, m_new, F, M, pts


def compute_boundary_curve(points, alpha=0.1):
    """Compute the convex hull boundary of 2D points."""
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        poly = alphashape.alphashape(points, alpha)
        if isinstance(poly, sgeom.MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)
        return np.array(poly.exterior.coords)
    except Exception:
        hull = sgeom.MultiPoint(points).convex_hull
        return np.array(hull.exterior.coords)


def make_figure(orig_xy, dense_xy, boundary_xy):
    """Create Plotly figure of the interpolated crane curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=orig_xy[:, 0],
        y=orig_xy[:, 1],
        mode="markers+lines",
        name="Original matrix points",
        marker=dict(color="blue", size=6),
        line=dict(dash="dot", color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=dense_xy[:, 0],
        y=dense_xy[:, 1],
        mode="markers",
        name="Interpolated points",
        marker=dict(color="orange", size=3, opacity=0.5),
    ))
    if len(boundary_xy) > 2:
        fig.add_trace(go.Scatter(
            x=boundary_xy[:, 0],
            y=boundary_xy[:, 1],
            mode="lines",
            name="Envelope curve",
            line=dict(color="red", width=2),
        ))

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Outreach [m]",
        yaxis_title="Height [m]",
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=40),
        height=700,
    )
    return fig


# ============================================================
# Dash / Flask Application
# ============================================================

def create_app():
    server = Flask(__name__)
    app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True, title="Crane Curve Viewer (Dash)")

    DATA_DIR = os.getenv("DATA_DIR", "./data")
    HEIGHT_CSV = os.getenv("HEIGHT_CSV", "height.csv")
    OUTREACH_CSV = os.getenv("OUTREACH_CSV", "outreach.csv")

    # Load data
    height_df = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    fold_angles, main_angles, height_itp, outre_itp = build_interpolators(height_df, outreach_df)
    orig_xy = np.column_stack([outreach_df.values.ravel(), height_df.values.ravel()])

    # Layout
    app.layout = html.Div([
        html.H2("Crane Curve Viewer (Dash • Flask)"),
        html.Div([
            html.Label("Step between Main-jib matrix angles (degrees)"),
            dcc.Slider(id="slider-main-step", min=0.1, max=10, step=0.1, value=1.0,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Step between Folding-jib matrix angles (degrees)"),
            dcc.Slider(id="slider-fold-step", min=0.1, max=10, step=0.1, value=1.0,
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"}),

        html.Div(id="angle-summary", style={"marginTop": "8px", "fontFamily": "monospace"}),
        dcc.Graph(id="graph", style={"height": "78vh"}),

        dcc.Store(id="orig-xy", data=orig_xy.tolist()),
        dcc.Store(id="fold-angles", data=fold_angles.tolist()),
        dcc.Store(id="main-angles", data=main_angles.tolist()),
    ], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "18px"})

    # Callback
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
        fnew, mnew, F, M, pts = resample_grid(np.array(fold_angles_store), np.array(main_angles_store),
                                              d_fold=fold_step_deg, d_main=main_step_deg)
        H_dense = height_itp(pts)
        R_dense = outre_itp(pts)
        dense_xy = np.column_stack([R_dense, H_dense])
        dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]
        boundary = compute_boundary_curve(dense_xy)
        fig = make_figure(np.array(orig_xy_store), dense_xy, boundary)
        summary = f"Interpolated grid → Main step: {main_step_deg:.2f}° | Folding step: {fold_step_deg:.2f}° | Points: {len(dense_xy)}"
        return fig, summary

    return app, server


# ============================================================
# Entrypoints for Render / Local
# ============================================================

app, server = create_app()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 8050)), debug=True)
