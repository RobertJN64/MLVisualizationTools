from typing import List, Dict
import pandas as pd
from os import path

#Backend functions and classes used by the other scripts

def colinfo(data: pd.DataFrame, exclude:List[str] = None) -> List[Dict]:
    """
    Helper function for generating column info dict for a datframe

    :param data: A pandas Dataframe
    :param exclude: A list of data items to exclude
    """
    if exclude is None:
        exclude = []

    coldata = []
    for item in data.columns:
        if item not in exclude:
            coldata.append({'name': item, 'mean': data[item].mean(),
                            'min': data[item].min(), 'max': data[item].max()})
    return coldata

def fileloader(target: str):
    """Specify a path relative to MLVisualizationTools"""
    return path.dirname(__file__) + '/' + target

def getTheme(theme, folder=None, figtemplate=None):
    """
    Backend function for loading theme css files.

    Theme can be 'light' or 'dark', and that will autoload the theme from dbc
    If folder is none, it is set based on the theme
    If figtemplate is none, it is set based on the theme

    Returns theme, folder

    :param theme: 'light' / 'dark' or a css url
    :param folder: path to assets folder
    :param figtemplate: Used for putting plotly in dark theme
    """
    import dash_bootstrap_components as dbc
    if theme == "light":
        theme = dbc.themes.FLATLY
        if folder is None:
            folder = fileloader('theme_assets/light_assets')
        if figtemplate is None:
            figtemplate = "plotly"

    elif theme == "dark":
        theme = dbc.themes.DARKLY
        if folder is None:
            folder = fileloader('theme_assets/dark_assets')
        if figtemplate is None:
            figtemplate = "plotly_dark"

    return theme, folder, figtemplate

def getDashApp(title:str, notebook:bool, usetunneling:bool, host:str, port:int, mode: str, theme, folder):
    """
    Creates a dash or jupyter dash app, returns the app and a function to run it

    :param title: Passed to dash app
    :param notebook: Uses jupyter dash with default port of 1005 (instead of 8050)
    :param usetunneling: Enables ngrok tunneling for kaggle notebooks
    :param host: Passed to dash app.run()
    :param port: Can be used to override default ports
    :param mode: Could be 'inline', 'external', or 'jupyterlab'
    :param theme: Passed to dash app external stylesheets
    :param folder: Passed to assets folder to load theme css
    """

    if port is None:
        if notebook:
            port = 1005
        else:
            port = 8050

    if notebook:
        if usetunneling:
            try:
                from pyngrok import ngrok
            except ImportError:
                raise ImportError("Pyngrok is required to run in a kaggle notebook. "
                                  "Use pip install MLVisualizationTools[kaggle-notebook]")
            ngrok.kill() #disconnects active tunnels
            tunnel = ngrok.connect(port)
            print("Running in an ngrok tunnel. This limits you to 40 requests per minute and one active app.",
                  "For full features use google colab instead.")
            url = tunnel.public_url
        else:
            url = None

        try:
            from jupyter_dash import JupyterDash
        except ImportError:
            raise ImportError("JupyterDash is required to run in a notebook. "
                              "Use pip install MLVisualizationTools[dash-notebook]")
        app = JupyterDash(__name__, title=title, server_url=url,
                               external_stylesheets=[theme], assets_folder=folder) #TODO - sever shutdown issues

        def f(*_): print("Not terminating server for args: ", _)
        app._terminate_server_for_port = f
    else:
        from dash import Dash
        app = Dash(__name__, title=title, external_stylesheets=[theme], assets_folder=folder)

    def runApp():
        if notebook:
            app.run_server(host=host, port=port, mode=mode, debug=True) #, use_reloader=False)
        else:
            app.run_server(host=host, port=port, debug=True) #, use_reloader=False)

    return app, runApp