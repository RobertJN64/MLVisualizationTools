import MLVisualizationTools as project
import pytest

def test_train_model():
    import MLVisualizationTools.examples.TrainTitanicModel

def test_run_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops aggressive error message printing
    from tensorflow import keras
    import pandas as pd

    model = keras.models.load_model('MLVisualizationTools/examples/Models/titanicmodel')
    df = pd.read_csv('MLVisualizationTools/examples/Datasets/Titanic/train.csv')

    # region preprocess
    header = list(df.columns)
    header.remove("Survived")

    X = df[header].values
    Y = df["Survived"].values

    _, accuracy = model.evaluate(X, Y)
    assert accuracy >= 0.75