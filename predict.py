from sklearn.externals import joblib
from settings import TRAINED_MODELS_PATH, IMAGE_ROWS, IMAGE_COLS
import os
import keras
from utils import load_image_keras

label_encoder = joblib.load(os.path.join(TRAINED_MODELS_PATH, 'label_encoder.scikitlearn'))
model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'cars-24-0.82.hdf5'))

back_side = '/home/i008/cars_train/00433.jpg'
front_image = 'back1.png'

im = load_image_keras(
    front_image,
    target_size=(IMAGE_ROWS, IMAGE_COLS)
)

im = im * 1.0 / 255
predictions = model.predict_classes(im)
print(label_encoder.inverse_transform(predictions))
