from dash import html
import dash
dash.register_page(__name__, path="/test", name="Test")

layout = html.Div([html.H3("Test Page"), html.P("Placeholder for future content.")], style={"padding": "16px"})
