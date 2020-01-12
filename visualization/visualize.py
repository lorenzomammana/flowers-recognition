from pathlib import Path
from keras import preprocessing
import sys
sys.path.append("..")
from keras_applications.efficientnet import swish
from vis.utils.utils import apply_modifications, find_layer_idx
import models
import numpy as np
import matplotlib.cm as cm
import os
import tensorflow as tf
from vis.visualization import visualize_saliency, overlay, visualize_cam
import matplotlib.pyplot as plt
from keras.activations import linear

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ARCHITECTURE = 'densenet'

root = Path('/home/ubuntu/flower-classification/flower_data_original/')
output_path = Path('/home/ubuntu/flower-output/')
train_path = root / 'train/'
valid_path = root / 'valid/'
test_path = root / 'test/'

model = models.densenet121()
model_name = 'densenet121'
TARGET_SIZE = 224

model.load_weights(output_path / '{}_acc.h5'.format(model_name))

# Avoid preprocessing just to plot the original image
test_datagen = preprocessing.image.ImageDataGenerator(
)

test_generator = test_datagen.flow_from_directory(directory=test_path,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  target_size=(TARGET_SIZE, TARGET_SIZE),
                                                  class_mode='categorical')

# Get first test image
orig_img, label = test_generator.__getitem__(0)

# Save it to file for debugging purposes
plt.imsave('output/saliency-input.png', orig_img.squeeze() / 255.0)

# Now we can apply the preprocessing function
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

output_idx = find_layer_idx(model, 'dense_1')

# Change output activation according to the documentation
model.layers[output_idx].activation = linear
model = apply_modifications(model, custom_objects={'swish': swish})

# Compute the saliency map
saliency_map = visualize_saliency(model, output_idx, [label], img, backprop_modifier=None)

plt.figure()
plt.imshow(saliency_map, cmap='jet')
plt.savefig('output/saliency.png')

plt.clf()

# Test the saliency map with the guided and relu modifier
for modifier in ['guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle(modifier)
    saliency_map = visualize_saliency(model, 3, [label], seed_input=img, backprop_modifier=modifier)
    plt.figure()
    plt.imshow(saliency_map, cmap='jet')
    plt.savefig('output/saliency_{}.png'.format(modifier))

    plt.clf()

# Grad-CAM requires the definition of the last convolutional layer
penultimate_layer = find_layer_idx(model, 'conv5_block16_2_conv')

for modifier in [None, 'guided', 'relu']:
    plt.figure()
    grads = visualize_cam(model, output_idx, [label],
                          seed_input=img, penultimate_layer_idx=penultimate_layer,
                          backprop_modifier=modifier)
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    plt.imshow(overlay(jet_heatmap, orig_img.squeeze()))

    if modifier is None:
        modifier = 'None'

    plt.savefig('output/gradcam_{}.png'.format(modifier))

    plt.clf()
