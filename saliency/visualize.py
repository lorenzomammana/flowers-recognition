from pathlib import Path
from keras import preprocessing
import sys
from keras_applications.efficientnet import swish
from vis.utils.utils import apply_modifications

import models
import numpy as np

import os
import tensorflow as tf
from vis.visualization import visualize_saliency
import matplotlib.pyplot as plt
from keras.activations import linear

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ARCHITECTURE = 'efficientnet'

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
    model_name = 'baseline'
    TARGET_SIZE = 224

model.load_weights(output_path / '{}_acc.h5'.format(model_name))

test_datagen = preprocessing.image.ImageDataGenerator(
    preprocessing_function=models.preprocessing,
)

test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                  class_mode='categorical')

img, label = test_generator.__getitem__(0)
label = np.argmax(label)
# pip install git+https://github.com/raghakot/keras-vis.git -U

model.layers[3].activation = linear
model = apply_modifications(model, custom_objects={'swish': swish})
saliency_map = visualize_saliency(model, 3, [label], img, backprop_modifier=None, grad_modifier="absolute")

plt.imsave(output_path / 'saliency.png', saliency_map)

test_datagen = preprocessing.image.ImageDataGenerator(
)

test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                  class_mode='categorical')

img, label = test_generator.__getitem__(0)
plt.imsave(output_path / 'saliency-input.png', img.squeeze() / 255.0)