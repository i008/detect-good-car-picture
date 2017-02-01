from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from settings import IMAGE_ROWS, IMAGE_COLS


def create_model(n_classes, img_rows=IMAGE_ROWS, img_cols=IMAGE_COLS):
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

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(346))
    model.add(Activation('relu'))

    model.add(Dense(221))
    model.add(Activation('relu'))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def create_fully_connected(shape_1d, n_classes):
    model = Sequential()
    model.add(Dense(512, input_dim=shape_1d))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(111))
    model.add(Activation('relu'))
    model.add(Dense(55))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model
