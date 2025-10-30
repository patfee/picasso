import dash
from dash import html, dcc, Input, Output
import numpy as np
import plotly.graph_objects as go

from app.shared.data import (
    get_context,
    resample_grid_by_factors,
    expand_angles_by_factor,
)
from app.shared.geom import compute_boundary_curve  # safe to keep if defined

# --------------------------------------------------------------------
# Dash Page Registration
# --------------------------------------------------------------------
dash.register_page(__name__, path="/harbour", name="Harbour Mode")

# --------------------------------------------------------------------
# Load context (shared interpolators, dataframes, constants)
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
                            options=[
                                {"label": f"{f}x", "value": f}
                                for f in [1, 2, 4, 8, 16]
                            ],
                            value=1,
                            clearable=False,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "30px"},
                ),
                html.Div(
                    [
                        html.Label("Main Jib Interpolation"),
                        dcc.Dropdown(
                            id="harbour-main-factor",
                            options=[
                                {"label": f"{f}x", "value": f}
                                for f in [1, 2, 4, 8, 16]
                            ],
                            value=1,
                            clearable=False,
                            style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block"},
                ),
            ],
            style={"marginBottom": "15px"},
        ),
        dcc.Graph(id="harbour-heatmap", style={"height": "85vh"}),
    ],
    style={"padding": "20px"},
)


# --------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------
@dash.callback(
    Output("harbour-heatmap", "figure"),
    Input("harbour-fold-factor", "value"),
    Input("harbour-main-factor", "value"),
)
def update_heatmap(fold_factor, main_factor):
    """
    Recompute interpolated grid & update heatmap.
    """
    # Generate extended grid (keeps original angle points)
    fnew, mnew, F, M, pts = resample_grid_by_factors(
        FOLD_ANGLES, MAIN_ANGLES, fold_factor, main_factor
    )

    # Interpolate load, height, and outreach on same grid
    load_vals = LOAD_ITP(pts)
    height_vals = HEIGHT_ITP(pts)
    outre_vals = OUTRE_ITP(pts)

    # reshape back to 2D grids
    LOAD = load_vals.reshape(F.shape)
    HEIGHT = height_vals.reshape(F.shape)
    OUTRE = outre_vals.reshape(F.shape)

    # We’ll use OUTRE on X axis, HEIGHT on Y, LOAD as color
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
        title=f"Interpolated Load Capacity Heatmap — Fold {fold_factor}x, Main {main_factor}x",
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
    )

    fig.update_yaxes(scaleanchor=None, scaleratio=1)  # free aspect ratio
    return fig
