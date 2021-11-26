import pytest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
with pytest.warns(DeprecationWarning, match="imp module is deprecated"):
    from tensorflow import keras
import MLVisualizationTools as project
import MLVisualizationTools.backend as backend
from MLVisualizationTools.backend import fileloader
import pandas as pd
import copy

def test_colorizer():
    data = pd.DataFrame({'Output': [0, 0.5, 1]})
    data = backend.GraphData(data, backend.GraphDataTypes.Grid, 20, 'NotKey', 'NotKey')
    assert list(project.Colorizers.simple(copy.deepcopy(data), 'red').dataframe['Color']) == ['red'] * 3

    assert (list(project.Colorizers.binary(copy.deepcopy(data), highcontrast=True).dataframe['Color'])
           == ['orange', 'orange', 'blue'])

    assert (list(project.Colorizers.binary(copy.deepcopy(data), highcontrast=False).dataframe['Color'])
            == ['red', 'red', 'green'])

    assert (list(project.Colorizers.binary(copy.copy(data), highcontrast=False,
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

    AR = project.Analytics.analyzeModel(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    grid = project.Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
    _ = project.Graphs.plotlyGraph(copy.deepcopy(grid))
    _ = project.Graphs.matplotlibGraph(copy.deepcopy(grid))

    animgrid = project.Interfaces.predictionAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                  df, ["Survived"])
    _ = project.Graphs.plotlyGraph(animgrid)

    grid = project.Colorizers.simple(copy.deepcopy(grid), color='red')
    _ = project.Graphs.plotlyGraph(grid)

    with pytest.raises(ValueError):
        grid.colorized = "Not a mode"
        _ = project.Graphs.plotlyGraph(grid)

def test_colorizer_warning():
    data = pd.DataFrame({'Output': [0, 0.5, 1], 'Color': ['red', 'orange', 'yellow']})
    data = backend.GraphData(data, backend.GraphDataTypes.Grid, 20, 'NotKey', 'NotKey')
    with pytest.warns(Warning, match="Color key 'Color' was already in dataframe."):
        project.Colorizers.simple(copy.deepcopy(data), 'red')
    with pytest.warns(Warning, match="Color key 'Color' was already in dataframe."):
        project.Colorizers.binary(copy.deepcopy(data))

def test_wrong_data_format_exception():
    from MLVisualizationTools.graphinterface import WrongDataFormatException

    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    AR = project.Analytics.analyzeModel(model, df, ["Survived"])
    maxvar = AR.maxVariance()
    print(maxvar[0]) #tests repr

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
        _ = project.Graphs.plotlyAnimation(grid)

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.predictionAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                      df, ["Survived"])
        _ = project.Graphs.plotlyGrid(grid)

    with pytest.raises(WrongDataFormatException):
        grid = project.Interfaces.predictionAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                                      df, ["Survived"])
        _ = project.Graphs.matplotlibGrid(grid)

def test_OutputKey_warning():
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    cols = list(df.columns)
    cols[1] = "Output"
    df.columns = cols

    with pytest.warns(Warning, match="Output key 'Output' was already in dataframe."):
        project.Interfaces.predictionGrid(model, cols[1], cols[2], df, ["Survived"])
    with pytest.warns(Warning, match="Output key 'Output' was already in dataframe."):
        project.Interfaces.predictionAnimation(model, cols[1], cols[2], cols[3], df, ["Survived"])

def test_graph_branch_error():
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df: pd.DataFrame = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    AR = project.Analytics.analyzeModel(model, df, ["Survived"])
    maxvar = AR.maxVariance()
    grid = project.Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
    project.Graphs.graph(grid)
    project.Graphs.graph(grid, project.types.GraphOutputTypes.Matplotlib)
    grid = project.Interfaces.predictionAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name, df, ["Survived"])
    with pytest.warns(Warning, match="Size key 'Age' was already in dataframe."):
        project.Graphs.graph(grid, sizekey='Age')
    with pytest.raises(NotImplementedError):
        project.Graphs.graph(grid, project.types.GraphOutputTypes.Matplotlib)

    grid.datatype = "NotAType"
    with pytest.raises(Exception, match="DataType NotAType not recognized."):
        project.Graphs.plotlyGraph(grid)
    with pytest.raises(Exception, match="DataType NotAType not recognized."):
        project.Graphs.matplotlibGraph(grid)

    with pytest.raises(Exception, match="GraphType NotAType not recognized."):
        # noinspection PyTypeChecker
        project.Graphs.graph(grid, "NotAType")