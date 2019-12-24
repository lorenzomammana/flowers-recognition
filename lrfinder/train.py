from pathlib import Path

import matplotlib

import sys

sys.path.append("..")

from patches.patch_loader import PatchLoader

matplotlib.use("Agg")

# import the necessary packages
from lrfinder.learningratefinder import LearningRateFinder
from lrfinder.clr_callback import CyclicLR
from lrfinder import config
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import sys
import tensorflow as tf
import models
from keras import preprocessing

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
                help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

PATCH_SIZE = 112

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)

root = Path('/home/ubuntu/blindness-dataset')
output_path = Path('/home/ubuntu/blindness-output/model_{}'.format(PATCH_SIZE))
train_dir_path = root / 'train_patches_{}'.format(PATCH_SIZE)
train_path = root / 'train_patches_{}.csv'.format(PATCH_SIZE)
test_dir_path = root / 'test_patches_{}'.format(PATCH_SIZE)
test_path = root / 'test_patches_{}.csv'.format(PATCH_SIZE)

loader = PatchLoader(train_path, test_path, 'png')

x_train = loader.only_train()

print("[INFO] compiling model...")

datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,  # randomly flip images
    brightness_range=[0.7, 1.3],
    preprocessing_function=models.preprocessing
)

print(config.MIN_LR)

opt = SGD(lr=config.MIN_LR, momentum=0.9)
model = models.patch_resnet18()

x_train['diagnosis'] = x_train['diagnosis'].astype('str')

train_generator = datagen.flow_from_dataframe(x_train,
                                              directory=train_dir_path,
                                              x_col='patch_id',
                                              y_col='diagnosis',
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=True,
                                              target_size=(PATCH_SIZE, PATCH_SIZE),
                                              class_mode='categorical')

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning
    # rates ranging from 1e-10 to 1e+1
    print("[INFO] finding learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(
        train_generator,
        1e-10, 1e+1,
        epochs=5,
        stepsPerEpoch=np.ceil((train_generator.n / float(config.BATCH_SIZE))),
        batchSize=config.BATCH_SIZE)

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

stepSize = config.STEP_SIZE * (train_generator.n // config.BATCH_SIZE)
clr = CyclicLR(
    mode=config.CLR_METHOD,
    base_lr=config.MIN_LR,
    max_lr=config.MAX_LR,
    step_size=stepSize)

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    train_generator,
    steps_per_epoch=np.ceil((train_generator.n / float(config.BATCH_SIZE))),
    epochs=config.NUM_EPOCHS,
    callbacks=[clr],
    verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")
