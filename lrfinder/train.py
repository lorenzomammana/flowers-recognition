from pathlib import Path

import matplotlib

import sys

sys.path.append("..")

matplotlib.use("Agg")

# import the necessary packages
from lrfinder.learningratefinder import LearningRateFinder
from lrfinder.clr_callback import CyclicLR
from lrfinder import config
from sklearn.metrics import classification_report
from keras.optimizers import SGD, adam
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import preprocess_crop
import cv2
import sys
import tensorflow as tf
import models
from keras import preprocessing
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
                help="whether or not to find optimal learning rate")
ap.add_argument('-b', type=int, default=64,
                help="batch size")
ap.add_argument('-t', type=int, default=224,
                help='target size')
args = vars(ap.parse_args())

BATCH_SIZE_TRAIN = args['b']
TARGET_SIZE = args['t']

print(TARGET_SIZE)

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

root = Path('/home/ubuntu/flower-classification/flower_data_original/')
output_path = Path('/home/ubuntu/flower-output/')
train_path = root / 'train/'
valid_path = root / 'valid/'
test_path = root / 'test/'


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

print("[INFO] compiling model...")

datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,  # randomly flip images
    brightness_range=[0.7, 1.3],
    rotation_range=10,
    preprocessing_function=models.preprocessing
)

opt = adam(lr=config.MIN_LR)
model = models.efficientnetb4()

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["acc"])

if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning
    # rates ranging from 1e-10 to 1e+1
    print("[INFO] finding learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(
        train_generator,
        config.MIN_LR, config.MAX_LR,
        epochs=15,
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


STEP_SIZE_TRAIN = np.ceil(train_generator.n / train_generator.batch_size)
STEP_SIZE_VALID = np.ceil(valid_generator.n / valid_generator.batch_size)

stepSize = config.STEP_SIZE * STEP_SIZE_TRAIN
clr = CyclicLR(
    mode=config.CLR_METHOD,
    base_lr=config.MIN_LR,
    max_lr=config.MAX_LR,
    step_size=stepSize)

# train the network
print("[INFO] training network...")

H = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=config.NUM_EPOCHS,
    callbacks=[clr],
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    workers=16,
    use_multiprocessing=False,
    max_queue_size=32,
    verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")

STEP_SIZE_TEST = np.ceil(test_generator.n / test_generator.batch_size)
predictions = model.predict_generator(test_generator,
                                      verbose=1,
                                      steps=STEP_SIZE_TEST,
                                      workers=16,
                                      use_multiprocessing=False,
                                      max_queue_size=32)

print(classification_report(test_generator.classes,
                            np.argmax(predictions, axis=1)))

# construct a plot that plots and saves the training history
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.TRAINING_PLOT_PATH)

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH)
