import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = Path("./data")
OUTREACH_CSV = DATA_DIR / "outreach.csv"
HEIGHT_CSV   = DATA_DIR / "height.csv"

TITLE = "Knuckleboom Coverage (Main Hoist) — Circle-based Angle Interpolation"

# ----------------------------
# Helpers
# ----------------------------

def _angles_from_headers(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expect:
      - Columns are main-jib angles in degrees (floats as header strings ok)
      - Index are folding-jib angles in degrees
    Returns (main_angles_deg, folding_angles_deg) as float arrays.
    """
    # columns can be strings, coerce to float
    main = np.array([float(c) for c in df.columns])
    # index too
    folding = np.array([float(i) for i in df.index])
    return main, folding


def _to_xy_matrices(outreach: pd.DataFrame, height: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two float matrices X (outreach) and Y (height) aligned on [folding, main]
    """
    # Ensure same ordering/shape
    outreach = outreach.sort_index().sort_index(axis=1)
    height   = height.reindex_like(outreach)

    X = outreach.values.astype(float)
    Y = height.values.astype(float)
    return X, Y


def _avg_radius_along_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    For each row (fixed folding angle), compute average radius sqrt(x^2+y^2).
    """
    R = np.sqrt(X*X + Y*Y)
    # robust average: median to reduce small tabular noise
    return np.median(R, axis=1)


@dataclass
class Circle:
    x0: float
    y0: float
    R: float

def _fit_circle_kasa(x: np.ndarray, y: np.ndarray) -> Circle:
    """
    Algebraic circle fit (Kåsa method). Works well for many points.
    Returns circle center and radius.
    """
    x = x.astype(float); y = y.astype(float)
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x*x + y*y
    # Solve least squares
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    R = np.sqrt(c + cx*cx + cy*cy)
    return Circle(cx, cy, R)


def _augment_axis_by_step(original_deg: np.ndarray, extra_step_deg: float) -> np.ndarray:
    """
    Between each pair of consecutive original angles a<b, insert intermediate
    angles at 'extra_step_deg'. Always include the original endpoints.
    """
    result: List[float] = []
    for a, b in zip(original_deg[:-1], original_deg[1:]):
        result.append(a)
        if extra_step_deg > 0:
            # number of interior points
            gap = b - a
            n = int(np.floor((gap / extra_step_deg))) - 1
            # Only insert when it makes sense
            if n >= 0:
                # generate interior points spaced by 'extra_step_deg', but never exceed b
                interior = a + np.arange(1, int(np.floor(gap/extra_step_deg))) * extra_step_deg
                # numerically safe clamp
                interior = interior[interior < b - 1e-9]
                result.extend(interior.tolist())
    result.append(original_deg[-1])
    return np.array(result, dtype=float)


def _interp_main_circle(
    main_deg_new: np.ndarray,
    main_deg_orig: np.ndarray,
    folding_deg_orig: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-wise interpolation along perfect circle centered at the pedestal (0,0).
    For each fixed folding row, compute an average radius; then compute x=R*cos(theta), y=R*sin(theta)
    for all theta in main_deg_new.
    Angles measured with 0° = horizontal, increasing counter-clockwise.
    """
    R_rows = _avg_radius_along_rows(X, Y)
    theta_new = np.deg2rad(main_deg_new)

    X1 = np.zeros((len(folding_deg_orig), len(main_deg_new)))
    Y1 = np.zeros_like(X1)

    for r_idx, R in enumerate(R_rows):
        X1[r_idx, :] = R * np.cos(theta_new)
        Y1[r_idx, :] = R * np.sin(theta_new)
    return X1, Y1


def _map_fold_angles_using_column_circles(
    folding_deg_new: np.ndarray,
    main_deg_new: np.ndarray,
    folding_deg_orig: np.ndarray,
    X1: np.ndarray,
    Y1: np.ndarray,
    X_orig: np.ndarray,
    Y_orig: np.ndarray,
    main_deg_orig: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Column-wise reparametrization for folding motion:
      For each main angle column (choose nearest available original column),
      fit a circle through the ORIGINAL tabular points vs folding angle.
      Use that circle to interpolate new folding angles by preserving the
      angular parameter around the fitted circle center.
    We then evaluate the circle at the interpolated parameter values.

    This respects the "folding rotates around main head" circle without
    needing explicit link lengths.
    """
    # For each new main angle, find the nearest original column (we use those data to fit a circle)
    nearest_col_indices = np.abs(main_deg_orig.reshape(-1,1) - main_deg_new.reshape(1,-1)).argmin(axis=0)
    X2 = np.zeros((len(folding_deg_new), len(main_deg_new)))
    Y2 = np.zeros_like(X2)

    for j_new, j_orig in enumerate(nearest_col_indices):
        # ORIGINAL column data at this main angle
        x_col = X_orig[:, j_orig]
        y_col = Y_orig[:, j_orig]

        # Fit a circle for the folding locus
        try:
            circ = _fit_circle_kasa(x_col, y_col)
        except Exception:
            # Fallback: keep pedestal-centered values for this column
            X2[:, j_new] = np.interp(folding_deg_new, folding_deg_orig, X1[:, j_new])
            Y2[:, j_new] = np.interp(folding_deg_new, folding_deg_orig, Y1[:, j_new])
            continue

        # Compute the "geometric angle" around this circle for each ORIGINAL folding sample
        phi_orig = np.arctan2(y_col - circ.y0, x_col - circ.x0)  # radians
        # unwrap for monotonic mapping
        phi_orig_unwrapped = np.unwrap(phi_orig)

        # Build a monotonic map fold_deg -> phi
        # (If fold degrees not strictly monotonic, sort together)
        sort_idx = np.argsort(folding_deg_orig)
        fdeg_sorted = folding_deg_orig[sort_idx]
        phi_sorted = phi_orig_unwrapped[sort_idx]

        # Interpolate phi for new folding degrees
        phi_new = np.interp(folding_deg_new, fdeg_sorted, phi_sorted)

        # Evaluate points on the fitted circle at phi_new
        X2[:, j_new] = circ.x0 + circ.R * np.cos(phi_new)
        Y2[:, j_new] = circ.y0 + circ.R * np.sin(phi_new)

    return X2, Y2


def make_plot(
    X_tab: np.ndarray, Y_tab: np.ndarray,
    X_dense: np.ndarray, Y_dense: np.ndarray,
    main_deg_dense: np.ndarray, folding_deg_dense: np.ndarray
) -> go.Figure:
    """
    Scatter of tabular points + outer envelope curve (dense grid convex hull).
    """
    fig = go.Figure()

    # Tabular points (diamond markers)
    fig.add_trace(go.Scatter(
        x=X_tab.ravel(), y=Y_tab.ravel(),
        mode="markers",
        name="Tabular points",
        marker=dict(symbol="diamond", size=6, opacity=0.8)
    ))

    # Draw a smoothed perimeter curve via convex hull on dense set
    pts = np.c_[X_dense.ravel(), Y_dense.ravel()]
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        loop = np.append(hull.vertices, hull.vertices[0])
        fig.add_trace(go.Scatter(
            x=pts[loop, 0], y=pts[loop, 1],
            mode="lines",
            name="Envelope",
            line=dict(width=2)
        ))
    except Exception:
        # fallback: thin-plate-ish line by sorting on angle
        ang = np.arctan2(pts[:,1], pts[:,0])
        order = np.argsort(ang)
        fig.add_trace(go.Scatter(
            x=pts[order,0], y=pts[order,1],
            mode="lines",
            name="Envelope (approx)",
            line=dict(width=2)
        ))

    fig.update_layout(
        title=TITLE,
        xaxis_title="Outreach [m]",
        yaxis_title="Jib head above pedestal flange [m]",
        xaxis=dict(constrain="domain", zeroline=True, showgrid=True),
        yaxis=dict(scaleanchor="x", scaleratio=1.0, zeroline=True, showgrid=True),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


# ----------------------------
# App
# ----------------------------

app = Dash(__name__)
server = app.server  # for gunicorn

# Load once (small files)
outreach_df = pd.read_csv(OUTREACH_CSV, index_col=0)
height_df   = pd.read_csv(HEIGHT_CSV, index_col=0)

main_deg_orig, folding_deg_orig = _angles_from_headers(outreach_df)
X_orig, Y_orig = _to_xy_matrices(outreach_df, height_df)

app.layout = html.Div([
    html.H2(TITLE, style={"margin":"8px 0 4px"}),
    html.Div([
        html.Div([
            html.Label("Extra resolution between ORIGINAL Main-jib matrix columns (degrees)"),
            dcc.Slider(
                id="main-step",
                min=0.0, max=5.0, step=0.25, value=1.0,
                marks={0:"0", 1:"1°", 2:"2°", 3:"3°", 4:"4°", 5:"5°"}
            ),
        ], style={"margin":"8px 12px"}),

        html.Div([
            html.Label("Extra resolution between ORIGINAL Folding-jib matrix rows (degrees)"),
            dcc.Slider(
                id="fold-step",
                min=0.0, max=5.0, step=0.25, value=1.0,
                marks={0:"0", 1:"1°", 2:"2°", 3:"3°", 4:"4°", 5:"5°"}
            ),
        ], style={"margin":"8px 12px"}),
    ], style={"display":"grid", "gridTemplateColumns":"1fr 1fr", "gap":"8px"}),

    html.Div(id="angles-info", style={"fontFamily":"monospace", "margin":"6px 12px"}),

    dcc.Graph(id="coverage-plot", style={"height":"78vh"}),
])


@app.callback(
    Output("coverage-plot", "figure"),
    Output("angles-info", "children"),
    Input("main-step", "value"),
    Input("fold-step", "value"),
)
def update(main_step_deg: float, fold_step_deg: float):
    # 1) MAIN interpolation (pedestal-centered circles)
    main_dense = _augment_axis_by_step(main_deg_orig, float(main_step_deg))
    X1, Y1 = _interp_main_circle(main_dense, main_deg_orig, folding_deg_orig, X_orig, Y_orig)

    # 2) FOLD interpolation (column circle fit around main head)
    fold_dense = _augment_axis_by
