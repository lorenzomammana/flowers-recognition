# Train the MLP using the cylcical learning rate

# import the necessary packages
import argparse
import sys

sys.path.append("..")

from lrfinder.clr_callback import CyclicLR
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os

from lrfinder.learningratefinder import LearningRateFinder
from numpy.random import seed
from tensorflow import set_random_seed

seed(42)
set_random_seed(42)


def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
    # open the input file for reading
    f = open(inputPath, "r")

    # loop indefinitely
    while True:
        # initialize our batch of data and labels
        data = []
        labels = []

        # keep looping until we reach our batch size
        while len(data) < bs:
            # attempt to read the next row of the CSV file
            row = f.readline()

            # check to see if the row is empty, indicating we have
            # reached the end of the file
            if row == "":
                # reset the file pointer to the beginning of the file
                # and re-read the row
                f.seek(0)
                row = f.readline()

                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break

            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")

            # update the data and label lists
            data.append(features)
            labels.append(label)

        # yield the batch to the calling function
        yield (np.array(data), np.array(labels))


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default=0,
                help="whether or not to find optimal learning rate")
ap.add_argument("-m", type=str, default='triangular',
                help="cyclical lr method")
ap.add_argument("-s", type=int, default=2,
                help="cyclical lr step size")
args = vars(ap.parse_args())

config.CLR_METHOD = args['m']
config.STEP_SIZE = args['s']
# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# derive the paths to the training, validation, and testing CSV files
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
                              "{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
                            "{}.csv".format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
                             "{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation
# sets
totalTrain = sum([1 for _ in open(trainPath)])
totalVal = sum([1 for _ in open(valPath)])

# extract the testing labels from the CSV file and then determine the
# number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
                                 config.CLASSES, mode="train")
valGen = csv_feature_generator(valPath, config.BATCH_SIZE,
                               config.CLASSES, mode="eval")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
                                config.CLASSES, mode="eval")

# define our simple neural network
model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * 512,), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(config.CLASSES, activation="softmax"))

# compile the model
opt = SGD(lr=config.MIN_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

if args["lr_find"] > 0:
    # initialize the learning rate finder and then train with learning
    # rates ranging from 1e-10 to 1e+1
    print("[INFO] finding learning rate...")
    lrf = LearningRateFinder(model)
    lrf.find(
        trainGen,
        1e-10, 1e-1,
        epochs=10,
        stepsPerEpoch=np.ceil(totalTrain / config.BATCH_SIZE),
        batchSize=config.BATCH_SIZE,
        use_multiprocessing=False)

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

stepSize = config.STEP_SIZE * np.ceil(totalTrain / config.BATCH_SIZE)

clr = CyclicLR(
    mode=config.CLR_METHOD,
    base_lr=config.MIN_LR,
    max_lr=config.MAX_LR,
    step_size=stepSize)

mcp_save_acc = ModelCheckpoint('output/temp_acc.h5',
                               save_best_only=True,
                               monitor='val_acc', mode='max')
mcp_save_loss = ModelCheckpoint('output/temp_loss.h5',
                                save_best_only=True,
                                monitor='val_loss', mode='min')
# train the network
print("[INFO] training network...")

H = model.fit_generator(
    trainGen,
    steps_per_epoch=np.ceil(totalTrain / config.BATCH_SIZE),
    epochs=config.NUM_EPOCHS,
    callbacks=[clr, mcp_save_loss, mcp_save_acc],
    validation_data=valGen,
    validation_steps=np.ceil(totalVal / config.BATCH_SIZE),
    verbose=1)

# evaluate the network and show a classification report
print("[INFO] evaluating network...")

model.load_weights('output/temp_acc.h5')
predIdxs = model.predict_generator(testGen,
                                   steps=np.ceil(totalTest / config.BATCH_SIZE))
predIdxs = np.argmax(predIdxs, axis=1)

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
plt.legend(loc="upper right")
plt.savefig(config.TRAINING_PLOT_PATH + 'acc_' + config.CLR_METHOD + '_' + str(config.STEP_SIZE) + '.png')

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH + config.CLR_METHOD + '_' + str(config.STEP_SIZE) + '.png')

accuracy = np.sum(predIdxs == testLabels) / totalTest

with open('output/results.txt', 'a') as f:
    f.write('Accuracy using best acc model with ' + config.CLR_METHOD + ' method and step-size ' +
            str(config.STEP_SIZE) + ': ' + str(accuracy))

model.load_weights('output/temp_loss.h5')
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
                                config.CLASSES, mode="eval")
predIdxs = model.predict_generator(testGen,
                                   steps=np.ceil(totalTest / config.BATCH_SIZE),
                                   verbose=1)
predIdxs = np.argmax(predIdxs, axis=1)

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
plt.legend(loc="upper right")
plt.savefig(config.TRAINING_PLOT_PATH + 'loss_' + config.CLR_METHOD + '_' + str(config.STEP_SIZE) + '.png')

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.CLR_PLOT_PATH + config.CLR_METHOD + '_' + str(config.STEP_SIZE) + '.png')

accuracy = np.sum(predIdxs == testLabels) / totalTest

with open('output/results.txt', 'a') as f:
    f.write('Accuracy using best loss model with ' + config.CLR_METHOD + ' method and step-size ' +
            str(config.STEP_SIZE) + ': ' + str(accuracy))
