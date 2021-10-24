#THIS EXAMPLE IS BROKEN AND IS CURRENTLY BEING BUGFIXED
raise NotImplementedError("THIS CODE IS CURRENTLY BROKEN")

from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers
from MLVisualizationTools.backend import fileloader
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #stops agressive error message printing
from tensorflow import keras

#todo - benchmarks

try:
    import plotly
except:
    raise ImportError("Plotly is required to run this demo. If you don't have plotly installed, install it with"
                      "`pip install plotly' or run the matplotlib demo instead.")

def main():
    model = keras.models.load_model(fileloader(__file__, 'Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader(__file__, 'Datasets/Titanic/train.csv'))

    AR = Analytics.Tensorflow(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    grid = Interfaces.TensorflowAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                     df, ["Survived"])
    grid = Colorizers.Binary(grid, highcontrast=True)
    print(grid)
    fig = Graphs.PlotlyAnimation(grid, maxvar[0].name, maxvar[1].name, maxvar[2].name)
    fig.show()
    with open('plotly.html', 'w+') as f:
        f.write(fig.to_html())

    # grid = Interfaces.TensorflowGrid(model, 'Parch', 'SibSp', #maxvar[0].name,
    #                                  df, ["Survived"], steps=10)
    # #grid = Colorizers.Binary(grid, highcontrast=True)
    # fig = Graphs.PlotlyGrid(grid, 'Parch', 'SibSp') # maxvar[0].name)
    # fig.show()

main()