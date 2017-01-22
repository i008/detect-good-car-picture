from keras.callbacks import ModelCheckpoint, History
from keras.preprocessing import image
import os
from models import create_model
from prepare_data import prepare_folder_structure
from settings import IMAGE_COLS, IMAGE_ROWS, TARGET_SIZE, TEST_PATH, TRAIN_PATH, TRAINED_MODELS_PATH, BALANCE

df_labels = prepare_folder_structure(
    minority_balanced=BALANCE
)

n_classes = df_labels.label.unique().shape[0]

model_dump_name = os.path.join(TRAINED_MODELS_PATH, "cars-{epoch:02d}-{val_acc:.2f}.hdf5")

checkpoint_callback = ModelCheckpoint(model_dump_name, save_best_only=True, monitor='val_acc')
history_callback = History()
callbacks = [checkpoint_callback, history_callback]

model = create_model(img_cols=IMAGE_COLS, img_rows=IMAGE_ROWS, n_classes=n_classes)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

imd_train = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True

)
imd_test = image.ImageDataGenerator(rescale=1. / 255)

imd_train_flow = imd_train.flow_from_directory(TRAIN_PATH,
                                               color_mode='rgb',
                                               target_size=TARGET_SIZE,
                                               batch_size=64,
                                               class_mode='categorical'
                                               )

imd_test_flow = imd_test.flow_from_directory(TEST_PATH,
                                             color_mode='rgb',
                                             batch_size=64,
                                             class_mode='categorical',
                                             target_size=TARGET_SIZE)

number_of_train_images = df_labels[df_labels.is_train].shape[0]

training_history = model.fit_generator(
    imd_train_flow,
    validation_data=imd_test_flow,
    samples_per_epoch=number_of_train_images * 3,
    nb_epoch=30,
    verbose=True,
    nb_val_samples=200,
    class_weight={0: 1, 1: 0.1808, 2: 0.38, 3: 0.03, 4: 0.2},
    callbacks=callbacks
)
