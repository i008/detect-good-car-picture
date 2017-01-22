import os

from sklearn.externals import joblib
import keras

from settings import TRAINED_MODELS_PATH, PROJECT_PATH

model_name = os.path.join(TRAINED_MODELS_PATH, 'cars-28-0.88.hdf5')
model_nn = keras.models.load_model(model_name)
label_encoder = joblib.load(os.path.join(PROJECT_PATH, 'trained_models/label_encoder.scikitlearn'))
