#THIS FILE IS NOT MEANT TO BE RUN AUTOMATICALLY BY TESTS
import time

class Benchmark:
    def __init__(self):
        self.timingtotal = 0
        self.activelabel = None
        self.starttime = -1
        self.timeinfo = {}

    def start(self, label):
        self.activelabel = label
        self.starttime = time.time()

    def stop(self, timePrint=True):
        s = time.time() - self.starttime
        if self.activelabel is None:
            raise Exception("Benchmark stopped with no active timer.")

        self.timeinfo[self.activelabel] = s
        if timePrint:
            print(self.activelabel, "finished in", round(s, 3), "seconds.")


    def next(self, label, timePrint=True):
        self.stop(timePrint)
        self.start(label)

benchmark = Benchmark()
benchmark.start("Pandas import")
import pandas as pd

benchmark.next("MLVisualization tools import")
from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers

benchmark.next("TF import")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
from tensorflow import keras

benchmark.stop()

start = time.time()
model = keras.models.load_model('MLVisualizationTools/examples/Models/titanicmodel')
print("Model load took: ", round(time.time() - start, 3), "seconds.")

start = time.time()
df = pd.read_csv('MLVisualizationTools/examples/Datasets/Titanic/train.csv')
print("CSV load took: ", round(time.time() - start, 3), "seconds.")

AR = Analytics.Tensorflow(model, df, ["Survived"])
maxvar = AR.maxVariance()

grid = Interfaces.TensorflowGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])
grid = Colorizers.Binary(grid)
fig = Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)
fig.show()

grid = Interfaces.TensorflowGrid(model, 'Parch', 'SibSp', df, ["Survived"])
grid = Colorizers.Binary(grid, highcontrast=True)
fig = Graphs.PlotlyGrid(grid, 'Parch', 'SibSp')
fig.show()