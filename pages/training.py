import dash
from dash import html, dcc, callback, Input, Output, State
import base64
import zipfile
from io import BytesIO


dash.register_page(__name__)


layout = html.Div([
    html.H1(children='Upload model Page'),
    dcc.Upload(
        id='upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-upload'),
])

#Funcionalidad para subir el modelo en zip y descomprimirlo
@callback(
    Output('output-upload', 'children'),
    [Input('upload', 'contents')],
    [State('upload', 'filename')]
)
def update_output(content, name):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        # If the file is a zip file, unzip it
        if name.endswith('.zip'):
            # Specify the directory where you want to save the file
            directory = './'
            z = zipfile.ZipFile(BytesIO(decoded))
            z.extractall(directory)
            return html.Div([
                f'Zip file {name} successfully uploaded and extracted.'
            ])
        else:
            return html.Div([
                'The uploaded file is not a .zip file.'
            ])


    return None
