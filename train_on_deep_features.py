from keras.utils.np_utils import to_categorical

from current_models import df_labels, optimizer_class_weights
from current_models import get_features
from current_models import label_encoder
from models import create_fully_connected

df_labels_balanced = df_labels
X_train = get_features(test=False)
X_test = get_features(test=True)
y_train = df_labels_balanced[df_labels_balanced.is_train]['label'].values
y_test = df_labels_balanced[~df_labels_balanced.is_train]['label'].values

y_train_one_hot = to_categorical(label_encoder.transform(y_train))
y_test_one_hot = to_categorical(label_encoder.transform(y_test))

# rfc = RandomForestClassifier(n_estimators=500,
#                              oob_score=True,
#                              n_jobs=-1)
# rfc.fit(X_train, y_train)
# print(rfc.oob_score_)



nn = create_fully_connected(X_train.shape[1], df_labels_balanced.label.unique().shape[0])
nn.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']

)

nn.fit(
    X_train,
    y_train_one_hot,
    nb_epoch=100,
    class_weight=optimizer_class_weights,
    validation_data=(X_test, y_test_one_hot)
)
