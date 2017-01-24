import os

from sklearn.externals import joblib
import keras

from settings import TRAINED_MODELS_PATH, PROJECT_PATH
import pandas as pd

model_nn = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'cars-28-0.88.hdf5'))
label_encoder = joblib.load(os.path.join(PROJECT_PATH, 'trained_models/label_encoder.scikitlearn'))
df_labels = pd.read_csv(os.path.join(TRAINED_MODELS_PATH, 'labels_df.csv'))
