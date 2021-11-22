# MLVisualizationTools

![Tests Badge](https://github.com/RobertJN64/MLVisualizationTools/actions/workflows/tests.yml/badge.svg)
![Python Version Badge](https://img.shields.io/pypi/pyversions/MLVisualizationTools)
![License Badge](https://img.shields.io/github/license/RobertJN64/MLVisualizationTools)

MLVisualizationTools is a python library to make
machine learning more understandable through the
use of effective visualizations.

We support graphing with matplotlib and plotly.
We implicity support all major ML libraries, such as 
tensorflow and sklearn.

You can use the built in apps to quickly anaylyze your
existing models, or build custom projects using the modular
sets of functions.

## Installation

`pip install MLVisualizationTools`

Depending on your use case, tensorflow, plotly and matplotlib might need to be
installed.

`pip install tensorflow`
`pip install plotly`
`pip install matplotlib`

To use interactive webapps, use the `pip install MLVisualizationTools[dash]` or `pip install MLVisualizationTools[dash-notebook]`
flags on install.

If you are running on a notebook that doesn't have dash support (like kaggle), you might need 
`pip install MLVisualizationTools[ngrok-tunneling]`

## Express

To get started using MLVisualizationTools, run one of the prebuilt apps.

```python
import MLVisualizationTools.express.DashModelVisualizer as App

model = ... #your keras model
data = ... #your pandas dataframe with features

App.visualize(model, data)
```

## Functions

MLVisualizationTools connects a variety of smaller functions.

Steps:
1. Keras Model and Dataframe with features
2. Analyzer
3. Interface / Interface Raw (if you don't have a dataframe)
4. Colorizers (optional)
5. Graphs

Analyzers take a keras model and return information about the inputs
such as which ones have high variance.

Interfaces take parameters and construct a multidimensional grid
of values based on plugging these numbers into the model.

(Raw interfaces allow you to use interfaces by specifying column
data instead of a pandas dataframe. Column data is a list with a dict with name, min,
max, and mean values for each feature column)

Colorizers mark points as being certain colors, typically above or below
0.5.

Graphs turn these output grids into a visual representation.

## Sample

```python
from MLVisualizationTools import Analytics, Interfaces, Graphs, Colorizers

#Displays plotly graphs with max variance inputs to model

model = ... #your model
df = ... #your dataframe
AR = Analytics.analyzeModel(model, df)
maxvar = AR.maxVariance()

grid = Interfaces.predictionGrid(model, maxvar[0].name, maxvar[1].name, df)
grid = Colorizers.binary(grid)
fig = Graphs.plotlyGraph(grid)
fig.show()
```

## Prebuilt Examples

Prebuilt examples run off of the pretrained model and dataset
packaged with this library. They include:
- Demo: a basic demo of library functionality that renders 2 plots
- MatplotlibDemo: Demo but with matplotlib instead of plotly
- DashDemo: Non-jupyter notebook version of an interactive dash
website demo
- DashNotebookDemo: Notebook version of an interactive website demo
- DashKaggleDemo: Notebook version of an dash demo that works in kaggle
notebooks

See [MLVisualizationTools/Examples](/MLVisualizationTools/examples) for more examples.
Use example.main() to run the examples and set parameters such as themes.

## Support for more ML Libraries

We support any ML library that has a `predict()` call that takes
a pd Dataframe with features. If this doesn't work, use a wrapper class like 
in this example:

```python
import pandas as pd

class ModelWrapper:
    def __init(self, model):
        self.model = model

    def predict(self, dataframe: pd.DataFrame):
        ... #Do whatever code you need here
```
