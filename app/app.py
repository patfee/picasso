import os
from pathlib import Path
import dash
import dash_bootstrap_components as dbc
from flask import Flask
from dash import html, dcc

# --- Environment variables / defaults ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
ASSETS_DIR = os.getenv("ASSETS_DIR", "./assets")
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")

# --- Define Flask + Dash app ---
server = Flask(__name__)

# Explicitly define pages folder path (important for clean import)
PAGES_DIR = str(Path(__file__).with_name("pages"))

app = dash.Dash(
    __name__,
    server=server,
    use_pages=True,
    pages_folder=PAGES_DIR,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="DCN Picasso DSV Engineering Data",
)

# --- Layout with simple sidebar navigation ---
sidebar = html.Div(
    [
        html.H3("Menu", style={"marginTop": 0}),
        dcc.Link("140T Main Hoist Envelope", href="/envelope", className="nav-link"),
        html.Br(),
        dcc.Link("Harbour Mode", href="/harbour", className="nav-link"),
        html.Br(),
        dcc.Link("TTS Data", href="/tts_data", className="nav-link"),
        html.Br(),
        dcc.Link("Test", href="/test", className="nav-link"),
    ],
    style={
        "position": "sticky",
        "top": "0px",
        "height": "100vh",
        "padding": "16px",
        "borderRight": "1px solid #e5e7eb",
        "background": "#fafafa",
    },
)

header = html.Div(
    [
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(
            src=LOGO_URL,
            style={
                "height": "42px",
                "objectFit": "contain",
                "position": "absolute",
                "right": "16px",
                "top": "12px",
            },
            alt="DCN Diving logo",
        ),
    ],
    style={"position": "relative", "padding": "12px 16px 4px 16px"},
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        header,
        html.Div(
            [
                html.Div(sidebar, style={"width": "260px", "flex": "0 0 260px"}),
                html.Div(dash.page_container, style={"flex": "1 1 auto"}),
            ],
            style={"display": "flex", "gap": "0px"},
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", 3000)), debug=True)
