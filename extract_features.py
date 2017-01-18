import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19

from utils import load_image_keras_imagenet_compatible


class ImageNetExtractor:
    def __init__(self, architecture='vgg19', include_top=False):
        self.architecture = architecture
        self.include_top = include_top

        if architecture == 'vgg19':
            self.model = VGG19(include_top=include_top)
        elif architecture == 'resnet':
            self.model = ResNet50(include_top=include_top)
        else:
            raise ValueError('unsupported architecture')

    def describe(self, array_of_images):
        shape = array_of_images.shape

        if not len(shape) == 4:
            raise ValueError('Keras required imnages to be passed as 4 dim array, for example shape = (n, 3, 222,222)')
        if not shape[1:] in [(3, 224, 224), (224, 224, 3)]:
            raise ValueError(
                'Imagenet requires images with shape (n, 3, 224, 224) / (n, 224, 224, 3) \n {}'.format(shape))

        return self.model.predict(array_of_images).flatten()

    def describe_from_path(self, list_of_image_pahts):
        array_of_images = np.concatenate([load_image_keras_imagenet_compatible(p) for p in list_of_image_pahts], axis=0)
        return self.describe(array_of_images)
