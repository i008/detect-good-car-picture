import os
from settings import TRAINED_MODELS_PATH
import keras

model_name = os.path.join(TRAINED_MODELS_PATH, 'cars-24-0.82.hdf5')
keras.models.load_model(model_name)


