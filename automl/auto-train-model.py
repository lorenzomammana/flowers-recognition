import csv
import os
from pathlib import Path
import autokeras as ak
import numpy as np

# root = Path('/home/ubuntu/flower-classification/flower_data_original/')
from automl.utils import load_image_dataset

autoroot = Path('/home/ubuntu/flower-classification/flower_data_autokeras/')
# output_path = Path('/home/ubuntu/flower-output/')
# train_path = root / 'train/'
# valid_path = root / 'valid/'
# test_path = root / 'test/'
#
# class_dirs = [i for i in os.listdir(path=train_path) if os.path.isdir(os.path.join(train_path, i))]
# with open(autoroot / 'labels-train.csv', 'w') as train_csv:
#     fieldnames = ['File Name', 'Label']
#     writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
#     writer.writeheader()
#     label = 0
#     for current_class in class_dirs:
#         for image in os.listdir(os.path.join(train_path, current_class)):
#             writer.writerow({'File Name': str(image), 'Label': label})
#         label += 1
#     train_csv.close()
#
# with open(autoroot / 'labels-valid.csv', 'w') as train_csv:
#     fieldnames = ['File Name', 'Label']
#     writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
#     writer.writeheader()
#     label = 0
#     for current_class in class_dirs:
#         for image in os.listdir(os.path.join(valid_path, current_class)):
#             writer.writerow({'File Name': str(image), 'Label': label})
#         label += 1
#     train_csv.close()
#
# with open(autoroot / 'labels-test.csv', 'w') as train_csv:
#     fieldnames = ['File Name', 'Label']
#     writer = csv.DictWriter(train_csv, fieldnames=fieldnames)
#     writer.writeheader()
#     label = 0
#     for current_class in class_dirs:
#         for image in os.listdir(os.path.join(test_path, current_class)):
#             writer.writerow({'File Name': str(image), 'Label': label})
#         label += 1
#     train_csv.close()

train_path = autoroot / 'train/'
valid_path = autoroot / 'valid/'
test_path = autoroot / 'test/'

x_train, y_train = load_image_dataset(csv_file_path=autoroot / "labels-train.csv",
                                      images_path=train_path, parallel=False)

print("Train generated")
x_valid, y_valid = load_image_dataset(csv_file_path=autoroot / "labels-valid.csv",
                                      images_path=valid_path, parallel=False)

print("Validation generated")
#
# x_test, y_test = load_image_dataset(csv_file_path=autoroot / "labels-test.csv",
#                                     images_path=test_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

input_node = ak.ImageInput()
output_node = ak.ImageBlock(
    # Normalize the dataset.
    normalize=False,
    # Do not do data augmentation.
    augment=True)(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=10)
clf.fit(x_train, y_train, validation_data=(x_valid, y_valid))
