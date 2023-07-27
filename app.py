import pandas as pd
import io
import dash
import numpy as np
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc



SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

if __name__ == '__main__':
# Definimos el menu lateral con sus links e iniciamos la aplicación
    app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        html.H1('VCF Variant Classificator', className="display-4"),
        html.Div(style={'height': '20px'}),

        html.Div(
            [
                html.Div([
                    dbc.Nav(
                        [
                            dbc.NavLink(page['name'], href=page["relative_path"]),
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ])
                for page in dash.page_registry.values()
            ], style=SIDEBAR_STYLE,
        ),
        dash.page_container
    ],style=CONTENT_STYLE)

    if __name__ == '__main__':
        app.run_server(debug=True)
