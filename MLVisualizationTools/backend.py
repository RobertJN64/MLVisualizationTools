from MLVisualizationTools.types import GraphDataTypes, ColorizerModes
from typing import List, Dict, Tuple
import pandas as pd
from os import path

#Backend functions and classes used by the other scripts

def colinfo(data: pd.DataFrame, exclude:List[str] = None) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Helper function for generating column info dict for a datframe

    :param data: A pandas Dataframe
    :param exclude: A list of data items to exclude
    """
    if exclude is None:
        exclude = []

    coldata = {}
    allcols = []
    for item in data.columns:
        if item not in exclude:
            coldata[item] = {'mean': data[item].mean(), 'min': data[item].min(), 'max': data[item].max()}
            allcols.append(item)
    return coldata, allcols

def fileloader(target: str, dynamic_model_version = True):
    """Specify a path relative to MLVisualizationTools"""
    if dynamic_model_version:
        if 'examples/Models' in target:
            import tensorflow as tf
            if float(tf.version.VERSION[:3]) < 2.5:
                target += "_v2.0"
    return path.dirname(__file__) + '/' + target

class GraphData:
    def __init__(self, dataframe: pd.DataFrame, datatype: GraphDataTypes, steps: int, x: str,
                 y: str, anim: str = None, outputkey: str = 'Output'):
        """Class for holding information about grid or animation data to be graphed."""
        self.dataframe = dataframe
        self.datatype = datatype

        self.colorized = ColorizerModes.NotColorized
        self.colorkey = None
        self.truecolor = None
        self.falsecolor = None

        self.truemsg = "Avg. Value is True"
        self.falsemsg = "Avg. Value is False"

        self.steps = steps

        self.x = x
        self.y = y
        self.anim = anim
        self.outputkey = outputkey

        self.dfdata = None #holds training data overlay

    def compileColorizedData(self):
        """
        Process a dataframe for use in a plotly graph.
        Returns a dataframe, a color key, a color_discrete_map, a category order, and a show legend bool
        """
        if self.colorized == ColorizerModes.NotColorized:
            return self.dataframe, None, None, None, False

        elif self.colorized == ColorizerModes.Simple:
            return self.dataframe, self.colorkey, None, None, False

        elif self.colorized == ColorizerModes.Binary:
            self.dataframe.loc[self.dataframe['Color'] == self.truecolor, 'Color'] = self.truemsg
            self.dataframe.loc[self.dataframe['Color'] == self.falsecolor, 'Color'] = self.falsemsg
            cdm = {self.truemsg: self.truecolor, self.falsemsg: self.falsecolor}
            co = {'Color': [self.truemsg, self.falsemsg]}
            return self.dataframe, self.colorkey, cdm, co, True

        else:
            raise ValueError(str(self.colorized) + " is not a valid colorizer mode.")
