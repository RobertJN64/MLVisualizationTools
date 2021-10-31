import pandas as pd
from MLVisualizationTools import Analytics, Interfaces
from MLVisualizationTools.modelanalytics import AnalyticsResult
from MLVisualizationTools.backend import fileloader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops agressive error message printing
from tensorflow import keras
import cProfile
model = keras.models.load_model(fileloader('examples/Models/titanicmodel'))
df = pd.read_csv(fileloader('examples/Datasets/Titanic/train.csv'))

AR = AnalyticsResult()
f = 'AR = Analytics.Tensorflow(model, df, ["Survived"])'
cProfile.run(f, 'analytics.prof')
maxvar = AR.maxVariance()

f = 'anim = Interfaces.TensorflowAnimation(model, maxvar[0].name, maxvar[1].name, maxvar[2].name, df, ["Survived"])'
cProfile.run(f, 'interface.prof')
if Interfaces == Analytics:
    pass