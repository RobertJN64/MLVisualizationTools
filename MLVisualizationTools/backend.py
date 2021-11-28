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
        self.color = None #holds color value from simple
        self.truecolor = None
        self.falsecolor = None

        self.truemsg = "Avg. Value is True"
        self.falsemsg = "Avg. Value is False"
        self.modelmessage = "Avg. Predictions from Model"
        self.datamessage = "Actual Data Values"

        self.steps = steps

        self.x = x
        self.y = y
        self.anim = anim
        self.outputkey = outputkey

        self.dfdata = None #holds training data overlay

    def should_show_key(self):
        if self.colorized == ColorizerModes.NotColorized:
            return False
        elif self.colorized == ColorizerModes.Simple:
            if self.dfdata is not None:
                return True
            else:
                return False
        elif self.colorized == ColorizerModes.Binary:
            return True
        else:
            raise ValueError(str(self.colorized) + " is not a valid colorizer mode.")

    def compileColorizedData(self, sizekey):
        """
        Process a dataframe for use in a plotly graph.
        Returns a dataframe, a color key, a color_discrete_map, and a category order
        """

        if self.colorized == ColorizerModes.NotColorized:
            return self.dataframe, None, None, None

        elif self.colorized == ColorizerModes.Simple:
            df = self.dataframe.copy()
            df[sizekey] = 5

            df[self.colorkey] = self.modelmessage
            cdm = {self.modelmessage: self.color}
            order = {self.colorkey: [self.color]}

            if self.dfdata is not None:
                self.dfdata['Color'] = self.datamessage
                dfdata = self.dfdata.copy()
                dfdata[sizekey] = dfdata[sizekey].apply(lambda x: x * 50 / dfdata[sizekey].max())
                df = df.append(dfdata)
                cdm[self.datamessage] = 'black'
                order[self.colorkey].append('black')

            return df, self.colorkey, cdm, order

        elif self.colorized == ColorizerModes.Binary:
            df = self.dataframe.copy()
            df[sizekey] = 5

            df.loc[df[self.colorkey] == self.truecolor, self.colorkey] = self.truemsg
            df.loc[df[self.colorkey] == self.falsecolor, self.colorkey] = self.falsemsg
            cdm = {self.truemsg: self.truecolor, self.falsemsg: self.falsecolor}
            order = {self.colorkey: [self.truemsg, self.falsemsg]}

            if self.dfdata is not None:
                self.dfdata['Color'] = self.datamessage
                dfdata = self.dfdata.copy()
                dfdata[sizekey] = dfdata[sizekey].apply(lambda x: x * 50/dfdata[sizekey].max())
                df = df.append(dfdata)
                cdm[self.datamessage] = 'black'
                order[self.colorkey].append('black')

            return df, self.colorkey, cdm, order

        else:
            raise ValueError(str(self.colorized) + " is not a valid colorizer mode.")
