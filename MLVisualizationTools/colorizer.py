import pandas as pd

def simpleColor(grid: pd.DataFrame, color):
    """Marks all points as being color specified"""
    grid['Color'] = [color] * len(grid)
    return grid
