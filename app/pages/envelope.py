from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import numpy as np

from app.shared.data import get_context, resample_grid_by_factors
from app.shared.data import compute_boundary_curve, _sample_points

# Register page
import dash
dash.register_page(__name__, path="/envelope", name="140T main hoist envelope")

CTX = get_context()

def factor_dropdown(id_, label, options, value):
    return html.Div([
        html.Label(label, style={"fontWeight": 600}),
        dcc.Dropdown(
            id=id_,
            options=[{"label": f"{o}× per-interval" + (" (original)" if o == 1 else ""), "value": o} for o in options],
            value=value, clearable=False, style={"marginBottom": "8px"}
        ),
    ], style={"marginBottom": "8px"})

controls = html.Div([
    factor_dropdown("main-factor", "Main-jib subdivision", [1, 2, 4, 8], 1),
    factor_dropdown("fold-factor", "Folding-jib subdivision", [1, 2, 4, 8, 16], 1),
    html.Div(id="main-range", style={"fontFamily": "monospace"}),
    html.Div(id="fold-range", style={"fontFamily": "monospace"}),
    dcc.Checklist(
        id="pedestal-toggle",
        options=[{"label": f"Include pedestal height (+{CTX['PEDESTAL_HEIGHT_M']:.1f} m)", "value": "on"}],
        value=["on"]
    ),
    html.Label("Envelope type", style={"fontWeight": 600}),
    dcc.RadioItems(
        id="envelope-type",
        options=[
            {"label": "None (points only)", "value": "none"},
            {"label": "Concave (fast approx)", "value": "concave"},
            {"label": "Convex hull (fast, stable)", "value": "convex"},
        ],
        value="convex", style={"marginTop": "4px"}
    ),
    html.Div(id="envelope-help", style={"fontSize": "12px", "color": "#555"}),
], style={"flex": "0 0 340px", "overflowY": "auto", "height": "78vh",
          "paddingRight": "12px", "borderRight": "1px solid #eee"})

graph = html.Div([dcc.Graph(id="graph", style={"height": "78vh"})],
                 style={"flex": "1 1 auto", "paddingLeft": "16px"})

layout = html.Div([controls, graph], style={"display": "flex", "padding": "16px", "gap": "10px"})

@callback(
    Output("main-range", "children"),
    Output("fold-range", "children"),
    Input("main-factor", "value"),
    Input("fold-factor", "value"),
)
def show_ranges(_m, _f):
    main_arr = CTX["main_angles"]
    fold_arr = CTX["fold_angles"]
    return (
        f"Main-jib angle range: {main_arr.min():.2f}° – {main_arr.max():.2f}° (original points: {len(np.unique(main_arr))})",
        f"Folding-jib angle range: {fold_arr.min():.2f}° – {fold_arr.max():.2f}° (original points: {len(np.unique(fold_arr))})",
    )

@callback(Output("envelope-help", "children"), Input("envelope-type", "value"))
def explain_envelope(kind):
    if kind == "none":
        return "Displays only points (no envelope). Useful for checking interpolation coverage."
    if kind == "concave":
        return "Concave alpha shape: tighter, realistic shape but sensitive to sparse data."
    return "Convex hull: stable and conservative outer boundary."

@callback(
    Output("graph", "figure"),
    Input("main-factor", "value"),
    Input("fold-factor", "value"),
    Input("pedestal-toggle", "value"),
    Input("envelope-type", "value"),
)
def update_figure(main_factor, fold_factor, pedestal_value, envelope_value):
    include_pedestal = "on" in (pedestal_value or [])
    prefer_concave = (envelope_value == "concave")
    draw_envelope = (envelope_value != "none")

    main_factor = int(main_factor or 1)
    fold_factor = int(fold_factor or 1)

    f = CTX["fold_angles"]
    m = CTX["main_angles"]
    _, _, _, _, pts = resample_grid_by_factors(f, m, fold_factor, main_factor)

    H_dense = CTX["height_itp"](pts)
    R_dense = CTX["outre_itp"](pts)
    if include_pedestal:
        H_dense = H_dense + CTX["PEDESTAL_HEIGHT_M"]

    dense_xy = np.column_stack([R_dense, H_dense])
    dense_xy = dense_xy[~np.isnan(dense_xy).any(axis=1)]

    envelope_xy = None
    if draw_envelope:
        envelope_xy = compute_boundary_curve(
            _sample_points(dense_xy, CTX["MAX_POINTS_FOR_ENVELOPE"]),
            prefer_concave=prefer_concave
        )

    fig = go.Figure()

    dense_for_plot = _sample_points(dense_xy, CTX["MAX_POINTS_FOR_DISPLAY"])
    if dense_for_plot.size:
        fig.add_trace(go.Scatter(
            x=dense_for_plot[:, 0], y=dense_for_plot[:, 1],
            mode="markers", marker=dict(size=4, symbol="diamond", opacity=0.5),
            name="Interpolated points",
            hovertemplate="Outreach: %{x:.2f} m<br>Height: %{y:.2f} m<extra></extra>"
        ))

    if envelope_xy is not None and len(envelope_xy) > 2:
        fig.add_trace(go.Scatter(
            x=envelope_xy[:, 0], y=envelope_xy[:, 1],
            mode="lines", line=dict(width=3), name="Envelope",
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
