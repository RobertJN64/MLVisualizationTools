#THIS FILE IS NOT MEANT TO BE RUN AUTOMATICALLY BY TESTS
import time

class Benchmark:
    def __init__(self):
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

    @staticmethod
    def pf(text, tofile, fname):
        print(text)
        if tofile:
            with open(fname, "a") as f:
                f.write(text + '\n')


    def printInfo(self, tofile=True, fname="benchmarkreport.txt"):
        if tofile:
            open(fname, 'w+') #wipes file
        print()
        self.pf("--- BENCHMARK REPORT ---", tofile, fname)
        total = sum(self.timeinfo.values())
        for item in self.timeinfo:
            s = item + " finished in " + str(round(self.timeinfo[item], 3)) + " seconds, "
            s += str(round(self.timeinfo[item] * 100 / total, 1)) + "% of total time."
            self.pf(s, tofile, fname)

def run_benchmark():
    benchmark = Benchmark()
    benchmark.start("Pandas import")
    import pandas as pd

    benchmark.next("MLVisualizationTools import")
    from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers
    from MLVisualizationTools.backend import fileloader

    benchmark.next("TF import")
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
    from tensorflow import keras

    benchmark.next("Model load")
    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))

    benchmark.next("CSV load")
    df = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    benchmark.next("Analytics call")
    AR = Analytics.analyzeModel(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    benchmark.next("Interface call")
    grid = Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name, df, ["Survived"])

    benchmark.next("Binary colorizer")
    grid = Colorizers.binary(grid)

    benchmark.next("Plotly fig creation")
    fig = Graphs.plotlyGraph(grid)

    benchmark.next("Plotly fig render")
    fig.show()

    benchmark.stop()
    benchmark.printInfo()

def resolution_compare():
    from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers
    from MLVisualizationTools.backend import fileloader
    import pandas as pd
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
    from tensorflow import keras

    model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
    df = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

    AR = Analytics.analyzeModel(model, df, ["Survived"])
    maxvar = AR.maxVariance()

    for steps in [5, 10, 15, 20, 50, 100]:
        start = time.time()
        grid = Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name,
                                         df, ["Survived"], steps)
        grid = Colorizers.binary(grid)
        end = time.time()
        print("Interface grid with", steps, "steps took", round(end-start, 3), "seconds to complete.")
        print("This is", round((end-start) / steps ** 2, 5), "seconds per item.")

        start = time.time()
        fig = Graphs.plotlyGraph(grid)
        end = time.time()
        print("Graph grid with", steps, "steps took", round(end - start, 3), "seconds to complete.")
        print("This is", round((end - start) / steps ** 2, 5), "seconds per item.")

        start = time.time()
        fig.show()
        end = time.time()
        print("Graph grid with", steps, "steps took", round(end - start, 3), "seconds to show.")
        print("This is", round((end - start) / steps ** 2, 5), "seconds per item.")

        print()

        start = time.time()
        anim = Interfaces.predictionAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name,
                                              df, ["Survived"], steps)
        anim = Colorizers.binary(anim)
        end = time.time()
        print("Animation grid with", steps, "steps took", round(end - start, 3), "seconds to complete.")
        print("This is", round((end - start) / steps ** 3, 5), "seconds per item.")

        start = time.time()
        fig = Graphs.plotlyGraph(anim)
        end = time.time()
        print("Animation graph with", steps, "steps took", round(end - start, 3), "seconds to complete.")
        print("This is", round((end - start) / steps ** 3, 5), "seconds per item.")

        start = time.time()
        fig.show()
        end = time.time()
        print("Animation graph with", steps, "steps took", round(end - start, 3), "seconds to show.")
        print("This is", round((end - start) / steps ** 3, 5), "seconds per item.")

        print()
        print()

if __name__ == "__main__":
    run_benchmark()
    #resolution_compare()