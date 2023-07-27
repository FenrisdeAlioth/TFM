import dash
from dash import html, dcc, callback, Input, Output, State, dash_table
import pandas as pd
import base64
import dash_bootstrap_components as dbc
import io
from dash.exceptions import PreventUpdate
import os
import tensorflow as tf
import numpy as np
import pickle

dash.register_page(__name__, path='/')

# Cargamos el modelo
try:
    model = tf.keras.models.load_model('variant_model_good_2')
except OSError:

    print('Failed to load the model.')


# Funcion para anotar el archivo de variatnes antes de la predicción
def run_vep():
    vep_executable = "~/ensembl-vep/vep"
    input_file = "./data.vcf"
    output_file = "/mnt/c/Users/administrador/PycharmProjects/pythonProject/TFG/test_p_an.vcf"
    cache_option = "--cache"
    merged_option = "--merged"
    af_gnomadg_option = "--af_gnomadg"
    sift_option = "--sift b"
    force_overwrite_option = "--force_overwrite"
    canonical_option = "--canonical"
    af_option = "--af"
    ambiguous_hgvs_option = "--ambiguous_hgvs"
    biotype_option = "--biotype"
    polyphen_option = "--polyphen b"
    fork_option = "--fork 10"
    tab_option = "--tab"

    command = (
        f"{vep_executable} "
        f"-i \"{input_file}\" "
        f"-o \"{output_file}\" "
        f"{cache_option} "
        f"{merged_option} "
        f"{af_gnomadg_option} "
        f"{sift_option} "
        f"{force_overwrite_option} "
        f"{canonical_option} "
        f"{af_option} "
        f"{ambiguous_hgvs_option} "
        f"{biotype_option} "
        f"{polyphen_option} "
        f"{fork_option} "
        f"{tab_option}"
    )

    os.system(command)


# Preprocesamiento del resultado de la anotación para la predicción
def read_vep_result():
    colnames = ['Uploaded_variation', 'Location', 'Allele', 'Gene', 'Feature', 'Feature_type', 'Consequence',
                'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons',
                'Existing_variation', 'IMPACT', 'DISTANCE', 'STRAND', 'FLAGS', 'BIOTYPE', 'CANONICAL', 'REFSEQ_MATCH',
                'SOURCE', 'REFSEQ_OFFSET', 'GIVEN_REF', 'USED_REF', 'BAM_EDIT', 'SIFT', 'PolyPhen', 'AF',
                'gnomADg_AF', 'gnomADg_AFR_AF', 'gnomADg_AMI_AF', 'gnomADg_AMR_AF', 'gnomADg_ASJ_AF', 'gnomADg_EAS_AF',
                'gnomADg_FIN_AF', 'gnomADg_MID_AF', 'gnomADg_NFE_AF', 'gnomADg_OTH_AF', 'gnomADg_SAS_AF', 'CLIN_SIG',
                'SOMATIC', 'PHENO']
    df_pred = pd.read_csv(f'test_p_an.vcf', sep='\t', comment='#', names=colnames, header=None, index_col=False)
    df_pred = df_pred[df_pred['gnomADg_AF'] != '-']
    columns_to_drop = ['Uploaded_variation', 'Location', 'cDNA_position', 'CDS_position', 'Protein_position', 'Gene',
                       'Feature', 'Feature_type',
                       'Existing_variation', 'DISTANCE', 'STRAND', 'CANONICAL', 'REFSEQ_MATCH',
                       'SOURCE', 'REFSEQ_OFFSET', 'GIVEN_REF', 'BAM_EDIT', 'PolyPhen', 'SIFT', 'AF', 'CLIN_SIG']
    df_pred = df_pred.drop(columns=columns_to_drop)
    columns_to_convert = ['gnomADg_AF', 'gnomADg_AFR_AF', 'gnomADg_AMI_AF', 'gnomADg_AMR_AF', 'gnomADg_ASJ_AF',
                          'gnomADg_EAS_AF',
                          'gnomADg_FIN_AF', 'gnomADg_MID_AF', 'gnomADg_NFE_AF', 'gnomADg_OTH_AF', 'gnomADg_SAS_AF']
    df_pred[columns_to_convert] = df_pred[columns_to_convert].replace('-', np.nan)
    df_pred[columns_to_convert] = df_pred[columns_to_convert].apply(pd.to_numeric)
    for c in columns_to_convert:
        columnas_para_media = [col for col in columns_to_convert if col not in [c]]
        media = df_pred[columnas_para_media].mean(
            axis=1)  # Calcula la media de las columnas excepto 'gnomADg_AF'
        df_pred[c].fillna(media, inplace=True)
    return df_pred


# Predicción
def pred_patho(df):
    try:
        model = tf.keras.models.load_model('variant_model_good_2')
    except OSError:
        print('Failed to load the model.')

    # Preprocesamiento de características categóricas
    categorical_features = ['Allele', 'Consequence', 'Amino_acids', 'Codons', 'IMPACT', 'FLAGS',
                            'BIOTYPE', 'USED_REF', 'SOMATIC', 'PHENO']

    categorical_features_2 = ['Allele', 'Consequence', 'Amino_acids', 'Codons', 'IMPACT', 'FLAGS',
                              'BIOTYPE', 'USED_REF', 'SOMATIC', 'PHENO', 'CLIN_SIG']
    # Preprocesamiento de características numéricas
    numerical_features = ['gnomADg_AF', 'gnomADg_AFR_AF', 'gnomADg_AMI_AF',
                          'gnomADg_AMR_AF', 'gnomADg_ASJ_AF', 'gnomADg_EAS_AF', 'gnomADg_FIN_AF',
                          'gnomADg_MID_AF', 'gnomADg_NFE_AF', 'gnomADg_OTH_AF', 'gnomADg_SAS_AF']
    encoders = {}
    for feature in categorical_features_2:
        with open(f'./encoders/{feature}_encoder.pkl', 'rb') as f:
            encoders[feature] = pickle.load(f)
    df['PHENO'] = df['PHENO'].astype(str)
    for feature in categorical_features:
        df[feature] = encoders[feature].transform(df[feature])
    y_pred_probs = model.predict([df[feature] for feature in categorical_features] + [
        df[numerical_features]])
    y_pred = np.argmax(y_pred_probs, axis=1)
    df['CLIN_SIG'] = y_pred
    for feature in categorical_features_2:
        df[feature] = encoders[feature].inverse_transform(df[feature])
    return df


# Preprocesamiento del archivo vcf cargado para eliminar columnas no necesarias
def clean_vcf(vcf_file):
    vcf_file = vcf_file.iloc[:, :7]
    return vcf_file


# lectura del vcf a dataframe
def load_vcf(vcf_file):
    # print('Archivo:' + vcf_file)
    vcf_df = pd.read_csv(vcf_file, delimiter='\t')
    # vcf_df_clean = clean_vcf(vcf_df)
    return vcf_df


# Una vez leido el archivo vcf por dash lo preprocesamos
def erase_symbol(decoded):
    lines = decoded.decode("utf-8").splitlines()  # Dividir el contenido en líneas

    filtered_lines = [line for line in lines if
                      not line.startswith("##")]  # Filtrar las líneas que no comienzan con "##"

    result = "\n".join(filtered_lines)  # Unir las líneas filtradas nuevamente en un solo string
    return result

# Layout web y callbacks de funcionalidad
layout = html.Div(children=[
    html.H1(children='Predict Page'),
    html.Hr(),
    html.Div(children=[
        dcc.Upload(
            id='upload-data',
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
            multiple=True
        ),
        html.Div(id='output-data-upload'),
    ])

])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    final_decoded = erase_symbol(decoded)
    df = None
    try:
        if 'vcf' in filename:
            df = load_vcf(io.StringIO(final_decoded))

    except Exception as e:
        # print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            id='data-predict',
        ),

        html.Hr(),  # horizontal line
        dbc.Button("Predict", color="primary", className="me-1", size="lg", id='pred-but', n_clicks=0),
        dbc.Button("Download CSV", color="primary", className="me-1", size="lg", id="btn_csv"),
        dbc.Button("Download JSON", color="primary", className="me-1", size="lg", id="btn_json"),
        dcc.Download(id="download"),

    ])


# Funcionalidad de los botones de descarga de csv y json
@callback(
    Output("download", "data"),
    [Input("btn_csv", "n_clicks"), Input("btn_json", "n_clicks")],
    [State('data-predict', 'data')],
    prevent_initial_call=True,
)
def func(n_clicks_csv, n_clicks_json, table_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    df = pd.DataFrame(table_data)
    if 'btn_csv' in changed_id:
        return dcc.send_data_frame(df.to_csv, "mydata.csv", index=False)
    elif 'btn_json' in changed_id:
        return dcc.send_data_frame(df.to_json, "mydata.json", orient="records")


#Funcionalidad del cuadro de upload
@callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


#Funcionalidad del boton de predicción
@callback(
    Output('data-predict', 'data'),
    Output('data-predict', 'columns'),
    Input('pred-but', 'n_clicks'),
    State('data-predict', 'data')
)
def update_button(n_clicks, data):
    if n_clicks == 0:
        raise PreventUpdate
    elif data is not None:

        df = pd.DataFrame(data)
        c_df = clean_vcf(df)
        c_df.to_csv("data.vcf", sep="\t", index=False)
        run_vep()
        df_p = read_vep_result()
        predicted_df = pred_patho(df_p)
        columns = [{"name": i, "id": i} for i in predicted_df.columns]
        # Return updated data and columns
        return predicted_df.to_dict('records'), columns
