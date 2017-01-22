import keras
import matplotlib.pyplot as plt
from pandas_ml.confusion_matrix import ConfusionMatrix
from sklearn.externals import joblib

from prepare_data import df_labels
from settings import *
from utils import load_images_keras

model_name = os.path.join(TRAINED_MODELS_PATH, 'cars-17-0.87.hdf5')
model_nn = keras.models.load_model(model_name)
label_encoder = joblib.load(os.path.join(PROJECT_PATH, 'trained_models/label_encoder.scikitlearn'))

df_test = df_labels[~df_labels.is_train]

test_images = load_images_keras(df_test.file_name.tolist(), normalize=True, target_size=(IMAGE_ROWS, IMAGE_COLS))
predictions = model_nn.predict_classes(test_images)
y_pred = label_encoder.inverse_transform(predictions)
y_true = df_test.label.tolist()

cm = ConfusionMatrix(y_true, y_pred)
print(cm.classification_report)
print(cm.F1_score)
cm.plot(normalized=True, backend='seaborn')
plt.show()
