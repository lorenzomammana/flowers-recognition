import csv
import os
from multiprocessing import Pool, cpu_count
import numpy as np
from skimage.transform import resize
import imageio


def read_csv_file(csv_file_path):
    """Read the csv file and returns two separate list containing file names and their labels.
    Args:
        csv_file_path: Path to the CSV file.
    Returns:
        file_names: List containing files names.
        file_label: List containing their respective labels.
    """
    file_names = []
    file_labels = []
    with open(csv_file_path, 'r') as files_path:
        path_list = csv.DictReader(files_path)
        fieldnames = path_list.fieldnames
        for path in path_list:
            file_names.append(path[fieldnames[0]])
            file_labels.append(path[fieldnames[1]])
    return file_names, file_labels


def read_image(img_path):
    """Read the image contained in the provided path `image_path`."""
    img = imageio.imread(uri=img_path)
    img = resize(img.astype('float32'), (400, 400), mode='reflect', anti_aliasing=True)
    return img


def _image_to_array(img_path):
    """Read the image from the path and return it as an numpy.ndarray.

    Load the image file as an array

    Args:
        img_path: a string whose value is the image file name
    """
    if os.path.exists(img_path):
        img = read_image(img_path)
        if len(img.shape) < 3:
            img = img[..., np.newaxis]
        return img
    else:
        raise ValueError("%s image does not exist" % img_path)


def read_images(img_file_names, images_dir_path, parallel=True):
    """Read the images from the path and return their numpy.ndarray instances.
    Args:
        img_file_names: List of strings representing image file names. # DEVELOPERS THERE'S PROBABLY A WAY TO MAKE THIS PARAM. OPTIONAL
        images_dir_path: Path to the directory containing images.
        parallel: (Default: True) Run _image_to_array will use multiprocessing.

    Returns:
        x_train: a list of numpy.ndarrays containing the loaded images.
    """
    img_paths = [os.path.join(images_dir_path, img_file)
                 for img_file in img_file_names]

    if os.path.isdir(images_dir_path):
        if parallel:
            pool = Pool(processes=cpu_count())
            x_train = pool.map(_image_to_array, img_paths)
            pool.close()
            pool.join()
        else:
            x_train = [_image_to_array(img_path) for img_path in img_paths]
    else:
        raise ValueError("Directory containing images does not exist")
    return np.asanyarray(x_train)


def load_image_dataset(csv_file_path, images_path, parallel=True):
    """Load images from their files and load their labels from a csv file.
    Assumes the dataset is a set of images and the labels are in a CSV file.
    The CSV file should contain two columns whose names are 'File Name' and 'Label'.
    The file names in the first column should match the file names of the images with extensions,
    e.g., .jpg, .png.
    The path to the CSV file should be passed through the `csv_file_path`.
    The path to the directory containing all the images should be passed through `image_path`.
    Args:
        csv_file_path: a string of the path to the CSV file
        images_path: a string of the path containing the directory of the images
        parallel: (Default: True) Load dataset using multiprocessing.
    Returns:
        x: Four dimensional numpy.ndarray. The channel dimension is the last dimension.
        y: a numpy.ndarray of the labels for the images
    """
    img_file_names, y = read_csv_file(csv_file_path)
    x = read_images(img_file_names, images_path, parallel)
    return np.array(x), np.array(y)
