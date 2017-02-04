import matplotlib.pyplot as plt
from pandas_ml.confusion_matrix import ConfusionMatrix

from current_models import model_nn, label_encoder, df_labels
from settings import *
from utils import load_images_keras

df_test = df_labels[~df_labels.is_train]

test_images = load_images_keras(df_test.file_name.tolist(), normalize=True, target_size=(IMAGE_ROWS, IMAGE_COLS))
predictions = model_nn.predict_classes(test_images)
pred_proba = model_nn.predict_proba(test_images)
y_pred = label_encoder.inverse_transform(predictions)
y_true = df_test.label.tolist()

cm = ConfusionMatrix(y_true, y_pred)
print(cm.classification_report)
print(cm.F1_score)
print(pred_proba)
cm.plot(normalized=True, backend='seaborn')
plt.show()


