import os

import keras
from sklearn.externals import joblib

from settings import TRAINED_MODELS_PATH, IMAGE_ROWS, IMAGE_COLS
from utils import load_image_keras

label_encoder = joblib.load(os.path.join(TRAINED_MODELS_PATH, 'label_encoder.scikitlearn'))
model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'cars-24-0.82.hdf5'))

# back_side =
front_image = '/home/i008/cars_train/00958.jpg'

im = load_image_keras(
    front_image,
    target_size=(IMAGE_ROWS, IMAGE_COLS)
)

im = im * 1.0 / 255
predict_class = model.predict_classes(im)
predict_proba = model.predict(im)
print(predict_proba)
print(label_encoder.inverse_transform(predict_class))
