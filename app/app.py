import os
from flask import Flask
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc  # optional; safe to keep even if unused

# Resolve assets folder (assumes project tree: ./assets next to ./app)
ASSETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))

server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    use_pages=True,               # Dash Pages
    assets_folder=ASSETS_PATH,    # serve from ../assets
    suppress_callback_exceptions=True,
    title="DCN Picasso DSV Engineering Data",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Left menu (auto-built from registered pages)
sidebar = html.Div(
    [
        html.H3("Menu", style={"marginTop": 0}),
        html.Div(
            [
                # order pages explicitly (path defines navigation)
                dcc.Link("140T main hoist envelope", href="/envelope", className="nav-link"),
                html.Br(),
                dcc.Link("Harbour Mode", href="/harbour", className="nav-link"),
                html.Br(),
                dcc.Link("Test", href="/test", className="nav-link"),
                html.Br(),
                dcc.Link("TTS Data", href="/tts", className="nav-link"),
            ]
        ),
    ],
    style={
        "position": "sticky", "top": "0px", "height": "100vh",
        "padding": "16px", "borderRight": "1px solid #e5e7eb", "background": "#fafafa",
        "width": "260px", "flex": "0 0 260px",
    },
)

# Header
LOGO_URL = os.getenv("LOGO_URL", "/assets/dcn_logo.svg")
header = html.Div(
    [
        html.H2("DCN Picasso DSV Engineering Data", style={"margin": 0}),
        html.Img(
            src=LOGO_URL,
            style={"height": "42px", "objectFit": "contain", "position": "absolute", "right": "16px", "top": "12px"},
            alt="DCN Diving logo",
        ),
    ],
    style={"position": "relative", "padding": "12px 16px 4px 16px"},
)

# Page container (Dash Pages renders the active page here)
content = html.Div(dash.page_container, style={"flex": "1 1 auto"})

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        header,
        html.Div(
            [
                sidebar,
                content,
            ],
            style={"display": "flex", "gap": "0px"},
        ),
    ]
)

if __name__ == "__main__":
    port = os.getenv("PORT")
    if not port or not port.isdigit():
        port = 3000
    else:
        port = int(port)
    app.run_server(host="0.0.0.0", port=port, debug=True)
