from MLVisualizationTools.backend import GraphData, ColorizerModes
import warnings

def simpleColor(data: GraphData, color) -> GraphData:
    """Marks all points as being the color inputted"""
    df = data.dataframe
    if 'Color' in df.columns:
        warnings.warn("Key 'Color' was already in dataframe. This could mean that 'Color' was a key in your "
                      "dataset or colorization has already been applied to the data. This could result in data "
                      "being overwritten.")
    df['Color'] = [color] * len(df)
    data.colorized = ColorizerModes.Simple
    return data

def binaryColor(data: GraphData, highcontrast:bool=True, truecolor=None, falsecolor=None,
                cutoff:float=0.5, outputkey: str = 'Output') -> GraphData:
    """
    Colors grid based on whether the value is higher than the cutoff. Default colors are green for true and red
    for false. Black will appear if an error occurs.

    :param data: Input data
    :param highcontrast: Switches default colors to blue for true and orange for false
    :param truecolor: Manually specify truecolor
    :param falsecolor: Manually specify falsecolor
    :param cutoff: Cutoff value, higher is true
    :param outputkey: Key to grab values from
    """
    df = data.dataframe
    if truecolor is None:
        if not highcontrast:
            truecolor = "green"
        else:
            truecolor = "blue"
    if falsecolor is None:
        if not highcontrast:
            falsecolor = "red"
        else:
            falsecolor = "orange"

    if 'Color' in df.columns:
        warnings.warn("Key 'Color' was already in dataframe. This could mean that 'Color' was a key in your "
                      "dataset or colorization has already been applied to the data. This could result in data "
                      "being overwritten.")

    df.loc[df[outputkey] > cutoff, 'Color'] = truecolor
    df.loc[df[outputkey] <= cutoff, 'Color'] = falsecolor
    data.colorized = ColorizerModes.Binary
    data.truecolor = truecolor
    data.falsecolor = falsecolor
    return data
