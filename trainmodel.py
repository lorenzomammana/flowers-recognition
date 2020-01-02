from pathlib import Path
from keras import preprocessing
import sys
import pandas as pd

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import models
import numpy as np
import preprocess_crop
import matplotlib.pyplot as plt
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

BATCH_SIZE_TRAIN = int(sys.argv[1])
ARCHITECTURE = sys.argv[2]

root = Path('/home/ubuntu/flower-classification/flower_data_original/')
output_path = Path('/home/ubuntu/flower-output/')
train_path = root / 'train/'
valid_path = root / 'valid/'
test_path = root / 'test/'

if ARCHITECTURE == 'densenet':
    model = models.densenet121()
    model_name = 'densenet121'
    TARGET_SIZE = 224
elif ARCHITECTURE == 'efficientnet':
    model = models.efficientnetb4()
    model_name = 'efficientnetb4'
    TARGET_SIZE = 380

train_datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,  # randomly flip images
    brightness_range=[0.7, 1.3],
    rotation_range=10,
    preprocessing_function=models.preprocessing
)

test_datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=models.preprocessing,
)

train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                    batch_size=BATCH_SIZE_TRAIN,
                                                    shuffle=True,
                                                    target_size=(TARGET_SIZE, TARGET_SIZE),
                                                    interpolation='lanczos:random',
                                                    class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(directory=valid_path,
                                                   batch_size=BATCH_SIZE_TRAIN,
                                                   shuffle=True,
                                                   target_size=(TARGET_SIZE, TARGET_SIZE),
                                                   class_mode='categorical')

test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                  class_mode='categorical')

if __name__ == '__main__':
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    mcp_save_acc = ModelCheckpoint((output_path / '{}_acc.h5'.format(model_name)).absolute().as_posix(),
                                   save_best_only=True,
                                   monitor='val_acc', mode='max')
    mcp_save_loss = ModelCheckpoint((output_path / '{}_loss.h5'.format(model_name)).absolute().as_posix(),
                                    save_best_only=True,
                                    monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    STEP_SIZE_TRAIN = np.ceil(train_generator.n / train_generator.batch_size)
    STEP_SIZE_VALID = np.ceil(valid_generator.n / valid_generator.batch_size)

    history = model.fit_generator(train_generator,
                                  epochs=50,
                                  verbose=1,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=[mcp_save_acc, mcp_save_loss, reduce_lr_loss, earlyStopping],
                                  workers=16,
                                  use_multiprocessing=False,
                                  max_queue_size=32)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(output_path / model_name / 'training_acc.png')

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(output_path / model_name / 'training_loss.png')

    model.load_weights(output_path / '{}_acc.h5'.format(model_name))

    STEP_SIZE_TEST = np.ceil(test_generator.n / test_generator.batch_size)
    predictions = model.predict_generator(test_generator,
                                          verbose=1,
                                          steps=STEP_SIZE_TEST,
                                          workers=16,
                                          use_multiprocessing=False,
                                          max_queue_size=32)

    accuracy = np.sum(np.argmax(predictions, axis=1) == test_generator.classes) / test_generator.samples

    print('Best acc model: {}'.format(accuracy))

    model.load_weights(output_path / '{}_loss.h5'.format(model_name))

    STEP_SIZE_TEST = np.ceil(test_generator.n / test_generator.batch_size)
    predictions = model.predict_generator(test_generator,
                                          verbose=1,
                                          steps=STEP_SIZE_TEST,
                                          workers=16,
                                          use_multiprocessing=False,
                                          max_queue_size=32)

    accuracy = np.sum(np.argmax(predictions, axis=1) == test_generator.classes) / test_generator.samples

    print('Best loss model: {}'.format(accuracy))
