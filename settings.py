import os

IMAGE_ROWS = 150
IMAGE_COLS = 150
TARGET_SIZE = (IMAGE_ROWS, IMAGE_COLS)
LABELS_FILE = 'labels.json'
EXP_NAME = 'exp'
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = '/home/i008'  # path where to create grouped image folders
FULL_EXP_PATH = os.path.join(BASE_PATH, EXP_NAME)
TRAIN_PATH = os.path.join(FULL_EXP_PATH, 'train')
TEST_PATH = os.path.join(FULL_EXP_PATH, 'test')
TRAINED_MODELS_PATH = os.path.join(PROJECT_PATH, 'trained_models')
IMAGES_PATH = '/home/i008/cars_train'  # path to images to train on
BALANCE = False
DEBUG = True
REVISION = '14:52'
