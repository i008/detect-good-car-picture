import os

import keras
from sklearn.externals import joblib

from settings import TRAINED_MODELS_PATH

model_name = os.path.join(TRAINED_MODELS_PATH, 'cars-24-0.82.hdf5')
model_nn = keras.models.load_model(model_name)
label_encoder = joblib.load('trained_models/label_encoder.scikitlearn')




