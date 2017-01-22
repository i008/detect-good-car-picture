import json
import os
import shutil

import pandas as pd
from i008.pandas_shortcuts import minority_balance_dataframe_by_multiple_categorical_variables
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from logger import logger
from settings import (
    LABELS_FILE, FULL_EXP_PATH, TRAIN_PATH, TEST_PATH, BALANCE, TRAINED_MODELS_PATH, IMAGES_PATH
)

EXCLUDE_LABELS = ['top', 'other', 'noclass']


def prepare_folder_structure(minority_balanced=None):
    print(TRAINED_MODELS_PATH)
    with open(LABELS_FILE, 'r') as labels:
        labels = json.loads(labels.read())
        labels = {os.path.join(IMAGES_PATH, k.split('/')[-1]): v for k, v in labels.items()}
        print(labels)

    df_labels = pd.DataFrame(data={'label': list(labels.values()), 'file_name': list(labels.keys())})
    df_labels = df_labels[~df_labels.label.isin(EXCLUDE_LABELS)]
    # df_labels = df_labels[df_labels.label.isin(['front-side','back-side'])]

    df_labels.set_value(df_labels.index, 'is_train', True)

    if minority_balanced:
        df_labels = minority_balance_dataframe_by_multiple_categorical_variables(df_labels,
                                                                                 categorical_columns=['label'])

    train, test = train_test_split(df_labels, test_size=0.2, stratify=df_labels.label)
    df_labels.set_value(train.index, 'is_train', True)
    df_labels.set_value(test.index, 'is_train', False)
    df_labels = df_labels.reindex_axis(sorted(df_labels.columns), axis=1)

    if os.path.exists(FULL_EXP_PATH):
        shutil.rmtree(FULL_EXP_PATH)

    os.makedirs(FULL_EXP_PATH)
    os.makedirs(TRAIN_PATH)
    os.makedirs(TEST_PATH)

    for label in df_labels.label.unique():
        os.makedirs(os.path.join(TRAIN_PATH, label))
        os.makedirs(os.path.join(TEST_PATH, label))

    for _, row in df_labels.iterrows():
        path, is_train, label = row[0], row[1], row[2]
        file_name = os.path.split(path)[-1]
        if is_train:
            copy_to = os.path.join(FULL_EXP_PATH, 'train', label, file_name)
            shutil.copyfile(path, copy_to)
        else:
            copy_to = os.path.join(FULL_EXP_PATH, 'test', label, file_name)
            shutil.copyfile(path, copy_to)

    logger.info(df_labels.label.value_counts())

    return df_labels


df_labels = prepare_folder_structure(minority_balanced=BALANCE)
label_encoder = LabelEncoder().fit(df_labels.label)
joblib.dump(label_encoder, os.path.join(TRAINED_MODELS_PATH, 'label_encoder.scikitlearn'))
