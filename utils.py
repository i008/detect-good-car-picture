import os

import numpy as np
from PIL import Image


def list_images(base_path, contains=None):
    return list_files(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)


def list_files(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath


def load_img(path, grayscale=False, target_size=None):
    """
    Load an image into PIL format.
    # Arguments
    path: path to image file
    grayscale: boolean
    target_size: None (default to original size)
    or (img_height, img_width)
    """

    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def load_image_keras(image_path, dim_ordering='default', gray=False, normalize=True, target_size=(300, 300)):
    from keras.preprocessing import image
    img = load_img(image_path, grayscale=gray, target_size=target_size)
    img = image.img_to_array(img, dim_ordering=dim_ordering)
    if normalize:
        img = img * 1 / 255
    return np.expand_dims(img, axis=0)


def load_images_keras(images_path_list, **kwargs):
    return np.concatenate(
        [load_image_keras(p, **kwargs) for p in images_path_list],
        axis=0
    )


def load_image_keras_imagenet_compatible(image_path, normalize_image=False, gray=False, target_size=(224, 224)):
    from keras.preprocessing import image
    from keras.applications.vgg19 import preprocess_input

    im = load_img(image_path, grayscale=gray, target_size=target_size)

    imarray = image.img_to_array(im)

    imarray = np.expand_dims(imarray, axis=0)
    imarray = preprocess_input(imarray)

    if normalize_image:
        imarray = imarray * 1 / 255

    return imarray

