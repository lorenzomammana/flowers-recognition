from pathlib import Path
from keras import preprocessing
import sys
import pandas as pd

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, SGD

import models
import numpy as np
import preprocess_crop
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from lrfinder.learningratefinder import LearningRateFinder
from lrfinder.clr_callback import CyclicLR

from numpy.random import seed
from tensorflow import set_random_seed

seed(42)
set_random_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

BATCH_SIZE_TRAIN = int(sys.argv[1])
ARCHITECTURE = sys.argv[2]
OPTIMIZER = sys.argv[3]
LR_FIND = int(sys.argv[4])
CROP = sys.argv[5]
METHOD = sys.argv[6]
STEP_SIZE = int(sys.argv[7])

if OPTIMIZER == 'adam':
    import lrfinder.configadam as config
else:
    import lrfinder.configsgd as config

config.STEP_SIZE = STEP_SIZE
config.CLR_METHOD = METHOD

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
else:
    model = models.resnet18()
    model_name = 'resnet18'
    TARGET_SIZE = 224

train_datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,  # randomly flip images
    brightness_range=[0.7, 1.3],  # decrease/increase luminosity
    rotation_range=10,  # rotation in range [-10,10]
    preprocessing_function=models.preprocessing
)

test_datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=models.preprocessing,
)

# Random cropping of dimension TARGET_SIZE x TARGET_SIZE
if CROP == 'random':
    train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                        batch_size=BATCH_SIZE_TRAIN,
                                                        shuffle=True,
                                                        target_size=(TARGET_SIZE, TARGET_SIZE),
                                                        interpolation='lanczos:random',
                                                        class_mode='categorical')
else:
    train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                        batch_size=BATCH_SIZE_TRAIN,
                                                        shuffle=True,
                                                        target_size=(TARGET_SIZE, TARGET_SIZE),
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
    if OPTIMIZER == 'adam':
        opt = Adam(lr=config.MIN_LR)
    else:
        opt = SGD(lr=config.MIN_LR, momentum=0.9)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    if LR_FIND == 1:
        # initialize the learning rate finder and then train with learning
        # rates ranging from 1e-10 to 1e+1
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find(
            train_generator,
            1e-10, 1e-1,
            epochs=10,
            stepsPerEpoch=np.ceil((train_generator.n / float(BATCH_SIZE_TRAIN))),
            batchSize=BATCH_SIZE_TRAIN)

        # plot the loss for the various learning rates and save the
        # resulting plot to disk
        lrf.plot_loss()
        plt.savefig(config.LRFIND_PLOT_PATH)

        # gracefully exit the script so we can adjust our learning rates
        # in the config and then train the network for our full set of
        # epochs
        print("[INFO] learning rate finder complete")
        print("[INFO] examine plot and adjust learning rates before training")
        sys.exit(0)

    # Save model if the validation loss decreases or the accuracy increases.
    mcp_save_acc = ModelCheckpoint((output_path / '{}_acc.h5'.format(model_name)).absolute().as_posix(),
                                   save_best_only=True,
                                   monitor='val_acc', mode='max')
    mcp_save_loss = ModelCheckpoint((output_path / '{}_loss.h5'.format(model_name)).absolute().as_posix(),
                                    save_best_only=True,
                                    monitor='val_loss', mode='min')

    STEP_SIZE_TRAIN = np.ceil(train_generator.n / train_generator.batch_size)
    STEP_SIZE_VALID = np.ceil(valid_generator.n / valid_generator.batch_size)

    # Define how many iterations are required to complete a learning rate cycle
    stepSize = config.STEP_SIZE * STEP_SIZE_TRAIN
    clr = CyclicLR(
        mode=config.CLR_METHOD,
        base_lr=config.MIN_LR,
        max_lr=config.MAX_LR,
        step_size=stepSize)

    # Using multiprocessing=True is unstable, but is generally faster!
    history = model.fit_generator(train_generator,
                                  epochs=50,
                                  verbose=1,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  callbacks=[clr, mcp_save_acc, mcp_save_loss],
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
    plt.savefig(output_path / model_name / 'training_acc_{}_{}_{}_{}.png'.format(OPTIMIZER, CROP, config.CLR_METHOD,
                                                                                 config.STEP_SIZE))

    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(output_path / model_name / 'training_loss_{}_{}_{}_{}.png'.format(OPTIMIZER, CROP, config.CLR_METHOD,
                                                                                  config.STEP_SIZE))

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

    # Save results
    with open('results.txt', 'a') as f:
        f.write('acc\t' + OPTIMIZER + '\t' + CROP + '\t' + config.CLR_METHOD + '\t' + str(config.STEP_SIZE) +
                '\t' + str(accuracy) + '\n')

    model.load_weights(output_path / '{}_loss.h5'.format(model_name))

    STEP_SIZE_TEST = np.ceil(test_generator.n / test_generator.batch_size)

    test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                      batch_size=BATCH_SIZE_TRAIN,
                                                      shuffle=False,
                                                      target_size=(TARGET_SIZE, TARGET_SIZE),
                                                      class_mode='categorical')

    predictions = model.predict_generator(test_generator,
                                          verbose=1,
                                          steps=STEP_SIZE_TEST,
                                          workers=16,
                                          use_multiprocessing=False,
                                          max_queue_size=32)

    accuracy = np.sum(np.argmax(predictions, axis=1) == test_generator.classes) / test_generator.samples

    with open('results.txt', 'a') as f:
        f.write('loss\t' + OPTIMIZER + '\t' + CROP + '\t' + config.CLR_METHOD + '\t' + str(config.STEP_SIZE) +
                '\t' + str(accuracy) + '\n')

    print('Best loss model: {}'.format(accuracy))
