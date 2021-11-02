# MLVisualizationTools

![Tests Badge](https://github.com/RobertJN64/MLVisualizationTools/actions/workflows/tests.yml/badge.svg)
![Python Version Badge](https://img.shields.io/pypi/pyversions/MLVisualizationTools)
![License Badge](https://img.shields.io/github/license/RobertJN64/MLVisualizationTools)

MLVisualizationTools is a python library to make
machine learning more understandable through the
use of effective visualizations.

It supports tensorflow, matplotlib, and plotly, with 
support for more ml libraries coming soon.

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

If you are running on a kaggle notebook, you might need 
`pip install MLVisualizationTools[kaggle-notebook]`

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
AR = Analytics.Tensorflow(model, df)
maxvar = AR.maxVariance()

grid = Interfaces.TensorflowGrid(model, maxvar[0].name, maxvar[1].name, df)
grid = Colorizers.Binary(grid)
fig = Graphs.PlotlyGrid(grid, maxvar[0].name, maxvar[1].name)
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
