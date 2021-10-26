import MLVisualizationTools as project
import pytest
import pandas as pd

def test_colorizer():
    data = pd.DataFrame({'feature': [0, 0.5, 1]})
    assert list(project.Colorizers.Simple(data.copy(), 'red')['Color']) == ['red'] * 3
    assert list(project.Colorizers.Binary(data.copy(), highcontrast=True)['Color']) == ['orange', 'orange', 'blue']
    assert list(project.Colorizers.Binary(data.copy(), highcontrast=False)['Color']) == ['red', 'red', 'green']

def test_demos():
    #no way to test dash demos yet
    import MLVisualizationTools.examples.Demo as Demo
    import MLVisualizationTools.examples.MatplotlibDemo as MPLDemo

    if Demo == MPLDemo: #clears warnings
        pass

def test_notimplemented():
    with(pytest.raises(NotImplementedError)):
        import MLVisualizationTools.examples.AnimationDemo as ADemo
        if ADemo == ADemo:
            pass

def test_train_model():
    import MLVisualizationTools.examples.TrainTitanicModel as TTM
    if TTM == TTM:
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
    assert accuracy >= 0.75