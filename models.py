from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential


def create_model(img_rows, img_cols, n_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(32))
    model.add(Activation('relu'))

    model.add(Dropout(0.3))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model
