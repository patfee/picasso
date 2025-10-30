from dash import html, dcc, callback, Output, Input, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ..shared.data import get_context, resample_grid_by_factors
from ..shared.geom import compute_boundary_curve

import dash
dash.register_page(__name__, path="/harbour", name="Harbour Mode")

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
    factor_dropdown("harbour-main-factor", "Main-jib subdivision", [1, 2, 4, 8], 1),
    factor_dropdown("harbour-fold-factor", "Folding-jib subdivision", [1, 2, 4, 8, 16], 1),
    html.Div(id="harbour-range", style={"fontFamily": "monospace", "color": "#444", "marginBottom": "8px"}),
    html.Button("Download dense Harbour table (CSV)", id="btn-harbour-csv",
                style={"width": "100%", "marginTop": "6px"}),
    dcc.Download(id="download-harbour-csv"),
    dcc.Store(id="harbour-dense-store"),
], style={"flex": "0 0 340px", "overflowY": "auto", "height": "88vh",
          "paddingRight": "12px", "borderRight": "1px solid #eee"})

graph = html.Div([dcc.Graph(id="harbour-heatmap", style={"height": "88vh"})],
                 style={"flex": "1 1 auto", "paddingLeft": "16px"})

layout = html.Div([
    html.H3("Harbour Mode (Cdyn 1.15)", style={"margin": "16px 16px 0 16px"}),
    html.Div([controls, graph], style={"display": "flex", "padding": "0 16px 16px 16px", "gap": "10px"}),
], style={"padding": "0"})

@callback(
    Output("harbour-dense-store", "data"),
    Output("harbour-range", "children"),
    Input("harbour-main-factor", "value"),
    Input("harbour-fold-factor", "value"),
)
def build_dense_harbour_table(main_factor, fold_factor):
    main_factor = int(main_factor or 1)
    fold_factor = int(fold_factor or 1)

    # Same per-interval subdivision (angle space)
    _, _, _, _, pts = resample_grid_by_factors(CTX["fold_angles"], CTX["main_angles"], fold_factor, main_factor)

    # Interpolate geometry + load at EXACT same points
    R_dense = CTX["outre_itp"](pts)      # Outreach [m]
    H_dense = CTX["height_itp"](pts)     # Height [m]
    Z_dense = CTX["load_itp"](pts)       # Load [t] Harbour Cdyn=1.15

    df = pd.DataFrame({
        "FoldingJib_deg": pts[:, 0],
        "MainJib_deg":    pts[:, 1],
        "Outreach_m":     R_dense,
        "Height_m":       H_dense,
        "Load_t":         Z_dense,
        "MainSubdivisionFactor": main_factor,
        "FoldingSubdivisionFactor": fold_factor,
    }).dropna()

    df = df.astype({
        "FoldingJib_deg": float, "MainJib_deg": float,
        "Outreach_m": float, "Height_m": float, "Load_t": float
    })

    msg = (f"Subdivisions — Main: {main_factor}× per-interval, "
           f"Folding: {fold_factor}× per-interval (rows: {len(df)})")
    return df.to_dict("records"), msg

@callback(
    Output("harbour-heatmap", "figure"),
    Input("harbour-dense-store", "data"),
)
def draw_harbour_from_dense(records):
    if not records:
        return go.Figure()

    df = pd.DataFrame(records)
    R = df["Outreach_m"].to_numpy()
    H = df["Height_m"].to_numpy()
    Z = df["Load_t"].to_numpy()

    r_lin = np.linspace(np.nanmin(R), np.nanmax(R), 300)
    h_lin = np.linspace(np.nanmin(H), np.nanmax(H), 300)
    RR, HH = np.meshgrid(r_lin, h_lin)

    Z_grid = griddata(np.column_stack([R, H]), Z, (RR, HH), method="linear")
    if np.isnan(Z_grid).any():
        Z_nn = griddata(np.column_stack([R, H]), Z, (RR, HH), method="nearest")
        Z_grid = np.where(np.isnan(Z_grid), Z_nn, Z_grid)

    # Envelope from same dense points
    env_xy = compute_boundary_curve(np.column_stack([R, H]), True)

    levels = dict(start=0, end=140, size=35, coloring="heatmap", showlines=False)
    colorscale = [
        [0.00, "#00b8ff"], [0.20, "#5ad5ff"], [0.40, "#20c997"],
        [0.60, "#b2df28"], [0.80, "#ffd84d"], [0.95, "#f39c12"], [1.00, "#c33b2b"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=r_lin, y=h_lin, z=Z_grid,
        contours=levels, colorscale=colorscale,
        colorbar=dict(title="Load [t]", tickmode="array", tickvals=[0, 35, 70, 105, 140]),
        hovertemplate="Outreach: %{x:.2f} m<br>Height: %{y:.2f} m<br>Load: %{z:.2f} t<extra></extra>",
        connectgaps=False
    ))

    if env_xy is not None and len(env_xy) > 2:
        fig.add_trace(go.Scatter(
            x=env_xy[:, 0], y=env_xy[:, 1],
            mode="lines", line=dict(color="#ffd84d", width=3),
            name="Envelope", hoverinfo="skip"
        ))
        # Mask outside for crisp edge
        try:
            import shapely.geometry as sgeom
            pts = np.column_stack([RR.ravel(), HH.ravel()])
            poly = sgeom.Polygon(env_xy)
            inside = np.array([poly.contains(sgeom.Point(xy)) for xy in pts]).reshape(RR.shape)
            fig.data[0].z = np.where(inside, Z_grid, np.nan)
        except Exception:
            pass

    fig.update_layout(
        title="Cdyn=1.15",
        template="plotly_dark",
        paper_bgcolor="#0b2733", plot_bgcolor="#0b2733",
        xaxis=dict(title="Outreach [m]", showgrid=True, gridcolor="#214a57", zeroline=False),
        yaxis=dict(title="Height [m]", showgrid=True, gridcolor="#214a57", zeroline=False),
        margin=dict(l=60, r=20, t=50, b=50),
        uirevision="keep", showlegend=False,
    )
    return fig

@callback(
    Output("download-harbour-csv", "data"),
    Input("btn-harbour-csv", "n_clicks"),
    State("harbour-dense-store", "data"),
    prevent_initial_call=True,
)
def download_harbour_dense(n, records):
    if not n or not records:
        raise dash.exceptions.PreventUpdate
    df = pd.DataFrame(records)
    return dcc.send_data_frame(df.to_csv, "harbour_dense_interpolated.csv", index=False)
