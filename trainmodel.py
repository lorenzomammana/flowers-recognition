from pathlib import Path
from keras import preprocessing
import sys
import models
import numpy as np
import tensorflow as tf
import preprocess_crop

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

BATCH_SIZE_TRAIN = 64
TARGET_SIZE = 224

root = Path('/home/ubuntu/flower-dataset/')
output_path = Path('/home/ubuntu/flower-output/')
train_path = root / 'train/'
valid_path = root / 'valid'
test_path = root / 'test'

train_datagen = preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,  # randomly flip images
    brightness_range=[0.7, 1.3],
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
                                                   interpolation='lanczos:center',
                                                   class_mode='categorical')

test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                  class_mode=None)

if __name__ == '__main__':
    model = models.build_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    STEP_SIZE_TRAIN = np.ceil(train_generator.n / train_generator.batch_size)
    STEP_SIZE_VALID = np.ceil(valid_generator.n / valid_generator.batch_size)

    model.fit_generator(train_generator,
                        epochs=20,
                        verbose=1,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        workers=16,
                        use_multiprocessing=False,
                        max_queue_size=32)
