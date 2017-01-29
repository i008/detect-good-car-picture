import os

import keras
import pandas as pd
from sklearn.externals import joblib

from settings import TRAINED_MODELS_PATH, PROJECT_PATH


def get_features(test=False):
    if test:
        file_name = 'vgg_features_test.numpy'
    else:
        file_name = 'vgg_features_train.numpy'

    return joblib.load(os.path.join(TRAINED_MODELS_PATH, file_name))


model_nn = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'cars-28-0.88.hdf5'))
label_encoder = joblib.load(os.path.join(PROJECT_PATH, 'trained_models/label_encoder.scikitlearn'))
df_labels = pd.read_csv(os.path.join(TRAINED_MODELS_PATH, 'labels_df.csv'))
# df_labels_balanced = pd.read_csv(os.path.join(TRAINED_MODELS_PATH, 'labels_df_balanced.csv'))


class_counts = df_labels.label.value_counts()
class_counts = min(class_counts) / class_counts
class_counts.index = class_counts.index.map(lambda x: label_encoder.transform([x])[0])
optimizer_class_weights = class_counts.to_dict()