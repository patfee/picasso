import dash
from dash import html, dcc, Input, Output
import numpy as np
import plotly.graph_objects as go

from app.shared.data import get_context, resample_grid_by_factors

# Optional import (only if geom.py exists — otherwise safe to keep)
try:
    from app.shared.geom import compute_boundary_curve
except ImportError:
    compute_boundary_curve = None

# --------------------------------------------------------------------
# Page Registration
# --------------------------------------------------------------------
dash.register_page(__name__, path="/harbour", name="Harbour Mode")

# --------------------------------------------------------------------
# Shared Data Context
# --------------------------------------------------------------------
CTX = get_context()
LOAD_ITP = CTX["load_itp"]
HEIGHT_ITP = CTX["height_itp"]
OUTRE_ITP = CTX["outre_itp"]
FOLD_ANGLES = CTX["fold_angles"]
MAIN_ANGLES = CTX["main_angles"]

# --------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------
layout = html.Div(
    [
        html.H3("Harbour Mode — Load Capacity Heatmap (Cdyn=1.15)"),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Folding Jib Interpolation"),
                        dcc.Dropdown(
                            id="harbour-fold-factor",
                            options=[{"label": f"{f}x", "value": f} for f in [1, 2, 4, 8, 16]],
                            value=1,
                            clearable=False,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "24px"},
                ),
                html.Div(
                    [
                        html.Label("Main Jib Interpolation"),
                        dcc.Dropdown(
                            id="harbour-main-factor",
                            options=[{"label": f"{f}x", "value": f} for f in [1, 2, 4, 8, 16]],
                            value=1,
                            clearable=False,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "24px"},
                ),
            ],
            style={"marginBottom": "15px"},
        ),

        dcc.Graph(id="harbour-heatmap", style={"height": "85vh"}),
    ],
    style={"padding": "20px"},
)

# --------------------------------------------------------------------
# Callback — updates the interpolated heatmap
# --------------------------------------------------------------------
@dash.callback(
    Output("harbour-heatmap", "figure"),
    Input("harbour-fold-factor", "value"),
    Input("harbour-main-factor", "value"),
)
def update_heatmap(fold_factor, main_factor):
    """
    Interpolate load matrix over expanded angle grids
    and plot as a smooth contour heatmap.
    """
    # Generate expanded angle grid
    fnew, mnew, F, M, pts = resample_grid_by_factors(
        FOLD_ANGLES, MAIN_ANGLES, fold_factor, main_factor
    )

    # Interpolate
    load_vals = LOAD_ITP(pts)
    height_vals = HEIGHT_ITP(pts)
    outre_vals = OUTRE_ITP(pts)

    # Reshape to 2D
    LOAD = load_vals.reshape(F.shape)
    HEIGHT = height_vals.reshape(F.shape)
    OUTRE = outre_vals.reshape(F.shape)

    # Build the contour plot
    fig = go.Figure(
        data=go.Contour(
            x=np.nanmean(OUTRE, axis=0),
            y=np.nanmean(HEIGHT, axis=1),
            z=LOAD,
            colorscale="Viridis",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            colorbar=dict(title="Load [t]", titleside="right"),
        )
    )

    fig.update_layout(
        xaxis_title="Outreach [m]",
        yaxis_title="Height [m]",
        title=f"Interpolated Load Capacity — Fold {fold_factor}x, Main {main_factor}x",
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
