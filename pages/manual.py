import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__)

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Manual de Usuario para la Aplicación de Análisis Genómico"),
                className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.H5("Descripción de la Aplicación"),
                className="mb-4"),
        html.Hr()
    ]),
    dbc.Row([
        dbc.Col(dcc.Markdown('''
        La aplicación de análisis genómico es una solución de software integral desarrollada para permitir el análisis de variantes genéticas a través de una interfaz web intuitiva y fácil de usar. Construida con Dash y Dash Bootstrap, la aplicación está diseñada para facilitar a los usuarios la carga, el análisis y la visualización de los datos de variantes genéticas, todo dentro de una única plataforma.
        
        El corazón de la aplicación es un algoritmo de machine learning que es capaz de identificar y clasificar variantes genéticas como benignas o patogénicas. Este algoritmo, basado en un conjunto de datos de variantes genéticas anotadas, ofrece un análisis detallado y preciso de los datos de variantes genéticas.
        
        La aplicación se divide en tres secciones principales:
        
        **Carga de Datos**: Los usuarios pueden cargar archivos VCF que contienen información sobre variantes genéticas. Los archivos se estandarizan al eliminar las columnas innecesarias, dejando sólo los datos esenciales para el análisis.
        
        **Análisis de Variantes**: Una vez cargados los datos, la aplicación anota y clasifica las variantes utilizando el algoritmo de machine learning. Los usuarios pueden ver los resultados en un formato de tabla fácilmente interpretable.
        
        **Visualización y Exportación de Resultados**: Los resultados del análisis se pueden visualizar dentro de la aplicación y también se pueden exportar en un formato de tabla descargable para un análisis posterior.
        
        Además, la aplicación incluye una función para reentrenar el algoritmo de machine learning. Esto permite a los usuarios cargar una base de datos de variantes en formato VCF anotado para que el algoritmo se reentrene y mejore su precisión a lo largo del tiempo.
        '''), className="mb-5"),
    ]),
    dbc.Row([
        dbc.Col(html.H5("Carga de datos"),
                className="mb-4")
    ]),html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Markdown('''
        La carga de datos en la aplicación es un proceso sencillo diseñado para ser intuitivo y fácil de usar.

        1. **Navega hasta el enlace "Predict"**: Este enlace se encuentra en la interfaz principal de la aplicación. Haz clic en él para acceder a la página de carga de datos.
        
        2. **Carga tu archivo VCF**: Verás un campo de subida de archivos en la página "Predict". Puedes cargar tu archivo VCF ya sea arrastrándolo y soltándolo en el campo, o haciendo clic en el campo y seleccionando el archivo de tu dispositivo.
        
        3. **Verifica tu archivo**: Una vez cargado, tu archivo se mostrará en la página para que puedas verificar que se ha cargado correctamente. Esto te permite asegurarte de que estás utilizando el archivo correcto antes de proceder.
               
        La calidad de las predicciones depende de la precisión de los datos que cargues. Por lo tanto, asegúrate de revisar cuidadosamente tus datos antes de iniciar el análisis.
        '''), className="mb-5"),
    ]),html.Hr(),
    dbc.Row([
        dbc.Col(html.H5("Anotación de Variantes y Clasificación"),
                className="mb-4")
    ]),html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Markdown('''
        Una vez que los datos se cargan en la aplicación, el proceso de anotación y clasificación de las variantes genéticas comienza. Aquí te explicamos cómo funciona este proceso:
        
        Cuando estés listo para comenzar el análisis, haz clic en el botón "Predict". Esto iniciará el algoritmo de machine learning que analizará tus datos y clasificará las variantes genéticas.
        
        1. **Anotación de Variantes**: La anotación es el primer paso que realizará el algoritmo. Durante esta fase, la aplicación utiliza Ensembl Variant Effect Predictor (VEP) para anotar las variantes genéticas presentes en tu archivo VCF. VEP añade información relevante como ubicación genómica, tipo de variante, efecto en genes y transcritos, y posibles implicaciones funcionales y clínicas. 
        
        2. **Clasificación de Variantes**: Después de la anotación, la aplicación clasifica las variantes como benignas o patogénicas utilizando un algoritmo de machine learning. Este algoritmo ha sido entrenado en un conjunto de variantes genéticas previamente anotadas, lo que le permite realizar predicciones precisas basadas en la información anotada de las variantes.
        
        
        
        La anotación y clasificación de las variantes son procesos automatizados que se llevan a cabo en el backend de la aplicación. Como usuario, sólo necesitas cargar tus datos y lanzar el análisis, y la aplicación se encargará del resto.
        '''), className="mb-5"),
    ]),html.Hr(),
    dbc.Row([
        dbc.Col(html.H5("Visualización y Exportación de Resultados"),
                className="mb-4")
    ]),html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Markdown('''
        Describe cómo los usuarios pueden visualizar y exportar los resultados del análisis.
        '''), className="mb-5"),
    ]),
],style={'textAlign': 'left'})