# USAGE
# python train.py

# import the necessary packages
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os

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
opt = SGD(lr=1e-2, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

earlyStopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

# train the network
print("[INFO] training simple network...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=np.ceil(totalTrain / config.BATCH_SIZE),
    validation_data=valGen,
    validation_steps=np.ceil(totalVal / config.BATCH_SIZE),
    epochs=50,
    callbacks=[earlyStopping, reduce_lr_loss]
)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability, then
# show a nicely formatted classification report
print("[INFO] evaluating network...")
predIdxs = model.predict_generator(testGen,
                                   steps=np.ceil(totalTest / config.BATCH_SIZE))
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
                            target_names=le.classes_))

accuracy = np.sum(predIdxs == testLabels) / totalTest

print(accuracy)
