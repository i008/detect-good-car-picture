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
    LABELS_FILE, FULL_EXP_PATH, TRAIN_PATH, TEST_PATH, TRAINED_MODELS_PATH, IMAGES_PATH
)

EXCLUDE_LABELS = ['top', 'other', 'noclass']


def prepare_folder_structure(minority_balanced=None):
    logger.info('Preparing folder structure')
    with open(LABELS_FILE, 'r') as labels:
        labels = json.loads(labels.read())
        labels = {os.path.join(IMAGES_PATH, k.split('/')[-1]): v for k, v in labels.items()}
        logger.debug(labels)

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

    le = LabelEncoder()
    le.fit(df_labels.label)
    joblib.dump(le, os.path.join(TRAINED_MODELS_PATH, 'label_encoder.scikitlearn'))

    logger.info(df_labels.label.value_counts())

    df_labels.to_csv(os.path.join(TRAINED_MODELS_PATH, 'labels_df.csv'))

    return df_labels
