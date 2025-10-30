# app/pages/envelope.py
import dash
from dash import html, dcc, Input, Output
import numpy as np
import plotly.graph_objects as go

from app.shared.data import get_context, resample_grid_by_factors
from app.shared.geom import compute_boundary_curve

dash.register_page(__name__, path="/envelope", name="Envelope")

CTX = get_context()
HEIGHT_ITP = CTX["height_itp"]
OUTRE_ITP  = CTX["outre_itp"]
LOAD_ITP   = CTX["load_itp"]
FOLD_ANGLES = CTX["fold_angles"]
MAIN_ANGLES = CTX["main_angles"]

layout = html.Div(
    [
        html.H3("Operating Envelope (Outreach–Height)"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Folding Jib Interpolation"),
                        dcc.Dropdown(
                            id="env-fold-factor",
                            options=[{"label": f"{f}x", "value": f} for f in [1, 2, 4, 8, 16]],
                            value=1, clearable=False, style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "24px"},
                ),
                html.Div(
                    [
                        html.Label("Main Jib Interpolation"),
                        dcc.Dropdown(
                            id="env-main-factor",
                            options=[{"label": f"{f}x", "value": f} for f in [1, 2, 4, 8, 16]],
                            value=1, clearable=False, style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "24px"},
                ),
                html.Div(
                    [
                        html.Label("Boundary mode"),
                        dcc.Dropdown(
                            id="env-boundary-mode",
                            options=[
                                {"label": "None (points only)", "value": "none"},
                                {"label": "Convex hull (fast, stable)", "value": "convex"},
                                {"label": "Concave (fast approx)", "value": "concave"},
                            ],
                            value="none", clearable=False, style={"width": "260px"},
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
            ],
            style={"marginBottom": "12px"},
        ),
        dcc.Graph(id="env-graph", style={"height": "85vh"}),
    ],
    style={"padding": "20px"},
)

@dash.callback(
    Output("env-graph", "figure"),
    Input("env-fold-factor", "value"),
    Input("env-main-factor", "value"),
    Input("env-boundary-mode", "value"),
)
def update_env(fold_factor, main_factor, boundary_mode):
    # Build the angle grid and sample
    fnew, mnew, F, M, pts = resample_grid_by_factors(FOLD_ANGLES, MAIN_ANGLES, fold_factor, main_factor)
    height_vals = HEIGHT_ITP(pts)
    outre_vals  = OUTRE_ITP(pts)
    load_vals   = LOAD_ITP(pts)

    # Convert to XY = (Outreach, Height)
    X = outre_vals
    Y = height_vals
    L = load_vals

    # Filter invalids
    mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(L)
    X, Y, L = X[mask], Y[mask], L[mask]

    fig = go.Figure()

    # Points (colored by load)
    fig.add_trace(
        go.Scattergl(
            x=X, y=Y, mode="markers",
            marker=dict(size=5, color=L, colorscale="Viridis", colorbar=dict(title="Load [t]")),
            name="Samples",
        )
    )

    # Optional boundary overlay
    if boundary_mode != "none":
        boundary = compute_boundary_curve(np.c_[X, Y], mode=boundary_mode, alpha=1.2)
        if boundary.size > 0:
            fig.add_trace(
                go.Scatter(
                    x=boundary[:, 0], y=boundary[:, 1],
                    mode="lines+markers", line=dict(width=2),
                    name=f"Boundary: {boundary_mode}",
                )
            )

    fig.update_layout(
        xaxis_title="Outreach [m]",
        yaxis_title="Height [m]",
        template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        title=f"Envelope — Fold {fold_factor}x, Main {main_factor}x, Boundary: {boundary_mode}",
    )
    return fig
