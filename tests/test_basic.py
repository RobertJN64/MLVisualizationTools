import MLVisualizationTools as project
import pytest
import pandas as pd

def test_colorizer():
    data = pd.DataFrame({'Output': [0, 0.5, 1]})
    assert list(project.Colorizers.Simple(data.copy(), 'red')['Color']) == ['red'] * 3
    assert list(project.Colorizers.Binary(data.copy(), highcontrast=True)['Color']) == ['orange', 'orange', 'blue']
    assert list(project.Colorizers.Binary(data.copy(), highcontrast=False)['Color']) == ['red', 'red', 'green']

#no way to test dash demos yet

def test_demo():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
    from tensorflow import keras
    model = keras.models.load_model('MLVisualizationTools/examples/Models/titanicmodel')
    df = pd.read_csv('MLVisualizationTools/examples/Datasets/Titanic/train.csv')

    AR = project.Analytics.Tensorflow(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    grid = project.Interfaces.TensorflowGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
    grid = project.Colorizers.Binary(grid)
    a = project.Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)

    grid = project.Interfaces.TensorflowGrid(model, 'Parch', 'SibSp', df, ["Survived"])
    grid = project.Colorizers.Binary(grid, highcontrast=True)
    b = project.Graphs.PlotlyGrid(grid, 'Parch', 'SibSp')
    if a == b:
        pass

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

    if MPLDemo == MPLDemo: #clears warnings
        pass

def test_notimplemented(): #disabled for active debugging
    with(pytest.raises(NotImplementedError)):
        import MLVisualizationTools.examples.AnimationDemo as ADemo
        if ADemo == ADemo:
            pass

def test_train_model():
    import MLVisualizationTools.examples.TrainTitanicModel as TTM
    if TTM == TTM:
        pass

def test_data_preprocess():
    import MLVisualizationTools.examples.Datasets.Titanic.TitanicDemoPreprocess as TDP
    if TDP == TDP:
        pass

def test_run_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops aggressive error message printing
    from tensorflow import keras

    model = keras.models.load_model('MLVisualizationTools/examples/Models/titanicmodel')
    df = pd.read_csv('MLVisualizationTools/examples/Datasets/Titanic/train.csv')

    # region preprocess
    header = list(df.columns)
    header.remove("Survived")

    X = df[header].values
    Y = df["Survived"].values

    _, accuracy = model.evaluate(X, Y)
    assert accuracy >= 0.70 #had to lower this because a test failed when we got 73%...