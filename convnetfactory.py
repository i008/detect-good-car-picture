from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


class ConvNetFactory:
    def __init__(self):
        pass

    @staticmethod
    def build(name, *args, **kargs):
        # define the network (i.e., string => function) mappings
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet": ConvNetFactory.KarpathyNet,
            "minivggnet": ConvNetFactory.MiniVGGNet,

        }

        # grab the builder function from the mappings dictionary
        builder = mappings.get(name, None)

        # if the builder is None, then there is not a function that can be used
        # to build to the network, so return None
        if builder is None:
            return None

        # otherwise, build the network architecture
        return builder(*args, **kargs)

    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        # initialzie the model
        model = Sequential()

        # define the first (and only) CONV => RELU layer
        model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation("relu"))

        # add a FC layer followed by the soft-max classifier
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        # return the network architecture
        return model

    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
        # initialize the model
        model = Sequential()
        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(20, 5, 5, border_mode="same", input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # return the network architecture
        return model

    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        model = Sequential()
        model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout:
            model.add(Dropout(0.25))
        model.add(Convolution2D(20, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout:
            model.add(Dropout(0.25))
        model.add(Convolution2D(20, 5, 5, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        if dropout:
            model.add(Dropout(0.25 * 2))
        model.add(Flatten())
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))

        # VGG ads multiple CONV->RELU prior applying destructive POOL to learn richer features
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        # number of filter rises
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        # dropout rises deepdown the network especially after FC layer.
        if dropout:
            model.add(Dropout(0.5))

        model.add(Dense(numClasses))
        model.add(Activation('softmax'))

        return model

    @staticmethod
    def Simple3conv(numChannels, imgRows, imgCols, numClasses, dropout=True, **kwargs):

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(numChannels, imgRows, imgCols)))
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

        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        return model

    @staticmethod
    def MiniVGGNet2(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(numChannels, imgRows, imgCols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation('relu'))

        # VGG ads multiple CONV->RELU prior applying destructive POOL to learn richer features
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        # number of filter rises
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if dropout:
            model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))

        # dropout rises deepdown the network especially after FC layer.
        if dropout:
            model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model


def VGG_16(weights_path=None, heatmap=False):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name="flatten"))
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, name='dense_3'))
    model.add(Activation("sigmoid", name="softmax"))

    return model
