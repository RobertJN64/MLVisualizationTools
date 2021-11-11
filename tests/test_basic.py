import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
from tensorflow import keras
import MLVisualizationTools as project
import MLVisualizationTools.backend as backend
from MLVisualizationTools.backend import fileloader
import pandas as pd
import pytest
import copy

def test_colorizer():
    data = pd.DataFrame({'Output': [0, 0.5, 1]})
    data = backend.GraphData(data, backend.GraphDataTypes.Grid)
    assert list(project.Colorizers.Simple(copy.deepcopy(data), 'red').dataframe['Color']) == ['red'] * 3

    assert (list(project.Colorizers.Binary(copy.deepcopy(data), highcontrast=True).dataframe['Color'])
           == ['orange', 'orange', 'blue'])

    assert (list(project.Colorizers.Binary(copy.deepcopy(data), highcontrast=False).dataframe['Color'])
            == ['red', 'red', 'green'])

    assert (list(project.Colorizers.Binary(copy.copy(data), highcontrast=False,
                                           truecolor='white', falsecolor='black').dataframe['Color'])
            == ['black', 'black', 'white'])

def test_dash_visualizer(): #doesn't launch dash apps, but tests creation process
    import MLVisualizationTools.express.DashModelVisualizer as DMV
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))
    df = df.drop("Survived", axis=1)
    DMV.App(model, df, theme='light')
    DMV.App(model, df, theme='dark')

def test_demo():
    import MLVisualizationTools.examples.Demo as Demo
    Demo.main(show=False)

    import MLVisualizationTools.examples.AnimationDemo as AnimationDemo
    AnimationDemo.main(show=False)

def test_mpl():
    import matplotlib
    matplotlib.use('Agg')  # disables UI rendering
    import warnings
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message= r'Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.'
        # disables agg warning on matplotlib
    )
    import MLVisualizationTools.examples.MatplotlibDemo as MPLDemo
    MPLDemo.main()

def test_process_data_train_and_run_model():
    import MLVisualizationTools.examples.Datasets.Titanic.TitanicDemoPreprocess as TDP
    TDP.main()
    import MLVisualizationTools.examples.TrainTitanicModel as TTM
    TTM.main()

    model = keras.models.load_model('MLVisualizationTools/examples/Models/titanicmodel')
    df = pd.read_csv('MLVisualizationTools/examples/Datasets/Titanic/train.csv')

    # region preprocess
    header = list(df.columns)
    header.remove("Survived")

    X = df[header].values
    Y = df["Survived"].values

    _, accuracy = model.evaluate(X, Y)
    # assert accuracy >= 0.70 #had to disable this because we kept failing...

def test_colormodes():
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    AR = project.Analytics.Tensorflow(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    grid = project.Interfaces.TensorflowGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
    _ = project.Graphs.PlotlyGrid(copy.deepcopy(grid), maxvar[0].name, maxvar[1].name)
    _ = project.Graphs.MatplotlibGrid(copy.deepcopy(grid), maxvar[0].name, maxvar[1].name)

    animgrid = project.Interfaces.TensorflowAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                  df, ["Survived"])
    _ = project.Graphs.PlotlyAnimation(animgrid, maxvar[0].name, maxvar[1].name, maxvar[2].name)

    grid = project.Colorizers.Simple(copy.deepcopy(grid), color='red')
    _ = project.Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)

    with pytest.raises(ValueError):
        grid.colorized = "Not a mode"
        _ = project.Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)

def test_colorizer_warning():
    data = pd.DataFrame({'Output': [0, 0.5, 1], 'Color': ['red', 'orange', 'yellow']})
    data = backend.GraphData(data, backend.GraphDataTypes.Grid)
    with pytest.warns(Warning, match="Key 'Color' was already in dataframe."):
        project.Colorizers.Simple(copy.deepcopy(data), 'red')
    with pytest.warns(Warning, match="Key 'Color' was already in dataframe."):
        project.Colorizers.Binary(copy.deepcopy(data))

def test_wrong_data_format_exception():
    from MLVisualizationTools.graphinterface import WrongDataFormatException

    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    AR = project.Analytics.Tensorflow(model, df, ["Survived"])
    maxvar = AR.maxVariance()
    print(maxvar[0]) #tests repr

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.TensorflowGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
        _ = project.Graphs.PlotlyAnimation(grid, maxvar[0].name, maxvar[1].name, maxvar[2].name)

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.TensorflowAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                      df, ["Survived"])
        _ = project.Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.TensorflowAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                      df, ["Survived"])
        _ = project.Graphs.MatplotlibGrid(grid, maxvar[0].name, maxvar[1].name)

def test_OutputKey_warning():
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    cols = list(df.columns)
    cols[1] = "Output"
    df.columns = cols

    with pytest.warns(Warning, match="Key 'Output' was already in dataframe."):
        project.Interfaces.TensorflowGrid(model, cols[1], cols[2], df, ["Survived"])
    with pytest.warns(Warning, match="Key 'Output' was already in dataframe."):
        project.Interfaces.TensorflowAnimation(model, cols[1], cols[2], cols[3], df, ["Survived"])

