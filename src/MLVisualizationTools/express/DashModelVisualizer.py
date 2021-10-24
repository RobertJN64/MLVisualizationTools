from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers
from MLVisualizationTools.backend import fileloader
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
from tensorflow import keras

try:
    import dash
    from dash import Input, Output
    from dash import dcc
    from dash import html
    import dash_bootstrap_components as dbc
    import plotly
except:
    raise ImportError("Dash and plotly are required to run this demo. Install them with the [dash] flag"
                      " on installation of this library.")

class App:
    def __init__(self, model, data: pd.DataFrame, title = "DashModelVisualizer", highcontrast=True,
                 notebook=False, inline=False, host='0.0.0.0', port=None):
        if notebook:
            from jupyter_dash import JupyterDash
            self.app = JupyterDash(__name__, title=title)
        else:
            self.app = dash.Dash(__name__, title=title)

        if port is None:
            if notebook:
                self.port = 1005
            else:
                self.port = 8050
        else:
            self.port = port

        self.model = model
        self.df = data
        self.highcontrast = highcontrast
        self.notebook = notebook
        self.inline = inline
        self.host = host

        options = []
        for col in self.df.columns:
            options.append({'label': col, 'value': col})

        self.AR = Analytics.Tensorflow(self.model, self.df)
        self.maxvar = self.AR.maxVariance()

        self.x = self.maxvar[0].name
        self.y = self.maxvar[1].name

        self.fig = self.updateGraph()

        graph = dbc.Card([
            dcc.Graph(id='example-graph', figure=self.fig)
        ], body=True)

        config = dbc.Card([
            dbc.Label("X Axis: "),
            dcc.Dropdown(id='xaxis', options=options, value=self.x),
            html.Br(),
            dbc.Label("Y Axis: "),
            dcc.Dropdown(id='yaxis', options=options, value=self.y),
            html.Br(),
        ], body=True)

        self.app.layout = dbc.Container([
            html.H1(title),
            html.Hr(),
            dbc.Row([
                dbc.Col(config, md=4),
                dbc.Col(graph, md=8)]
            ),
            html.P()])

        inputs = [Input('xaxis', "value"), Input('yaxis', 'value')]
        self.app.callback(Output("example-graph", "figure"), inputs)(self.updateGraphFromWebsite)

    def run(self):
        self.app.run_server(host = self.host, port = self.port)

    def updateGraph(self):
        data = Interfaces.TensorflowGrid(self.model, self.x, self.y, self.df)
        data = Colorizers.Binary(data, highcontrast=self.highcontrast)
        self.fig = Graphs.PlotlyGrid(data, self.x, self.y)
        return self.fig

    def updateGraphFromWebsite(self, x, y):
        self.x = x
        self.y = y
        return self.updateGraph()

def main(model, data: pd.DataFrame, title = "DashModelVisualizer", highcontrast=True, notebook=False,
         inline=False, host='0.0.0.0', port=None):
    """
    Creates a dash website to visualize an ML model.

    :param model: A tensorflow keras model
    :param data: A pandas dataframe, all df columns must be numerical model inputs
    :param title: Title for website
    :param highcontrast: Visualizes the model with orange and blue instead of green and red. Great for colorblind people!
    :param notebook: Uses jupyter dash instead of dash
    :param inline: If running in a notebook, whether or not to launch an external website
    :param host: default hostname for dash
    :param port: None for default port (8050) or (1005)
    """

    App(model, data, title, highcontrast, notebook, inline, host, port).run()

def default():
    model = keras.models.load_model(fileloader(__file__, 'Models/titanicmodel'))
    df = pd.read_csv(fileloader(__file__, 'Datasets/Titanic/train.csv'))
    df = df.drop('Survived', axis=1)
    main(model, df)

if __name__ == '__main__':
    default()