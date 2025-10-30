from dash import html, dcc, callback, Output, Input
import os

import dash
dash.register_page(__name__, path="/tts", name="TTS Data")

ASSETS_DIR = os.getenv("ASSETS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets")))

layout = html.Div([
    html.H3("TTS Data", style={"marginTop": 0}),
    html.P("Download the TTS crane GA and load charts PDFs below."),
    html.Div([
        html.Button("Download: Picasso_TTS_140_Tons_CraneGA.pdf", id="btn-tts-ga", n_clicks=0,
                    style={"marginRight": "8px", "marginBottom": "8px"}),
        dcc.Download(id="download-tts-ga"),
    ]),
    html.Div([
        html.Button("Download: 140Tons_TTS_Crane_load_charts.pdf", id="btn-tts-charts", n_clicks=0,
                    style={"marginRight": "8px", "marginBottom": "8px"}),
        dcc.Download(id="download-tts-charts"),
    ]),
    html.Hr(),
    html.Ul([
        html.Li(html.A("Open GA (inline)", href="/assets/Picasso_TTS_140_Tons_CraneGA.pdf", target="_blank")),
        html.Li(html.A("Open Load Charts (inline)", href="/assets/140Tons_TTS_Crane_load_charts.pdf", target="_blank")),
    ]),
], style={"padding": "16px"})

@callback(Output("download-tts-ga", "data"), Input("btn-tts-ga", "n_clicks"), prevent_initial_call=True)
def download_tts_ga(_):
    path = os.path.join(ASSETS_DIR, "Picasso_TTS_140_Tons_CraneGA.pdf")
    with open(path, "rb") as f:
        data = f.read()
    return dcc.send_bytes(lambda b: b.write(data), filename="Picasso_TTS_140_Tons_CraneGA.pdf")

@callback(Output("download-tts-charts", "data"), Input("btn-tts-charts", "n_clicks"), prevent_initial_call=True)
def download_tts_charts(_):
    path = os.path.join(ASSETS_DIR, "140Tons_TTS_Crane_load_charts.pdf")
    with open(path, "rb") as f:
        data = f.read()
    return dcc.send_bytes(lambda b: b.write(data), filename="140Tons_TTS_Crane_load_charts.pdf")
