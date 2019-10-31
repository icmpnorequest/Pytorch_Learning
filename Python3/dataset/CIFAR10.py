# coding=utf8
"""
@author: Yantong Lai
@date: 08/21/2019
"""

import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pandas as pd

# np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)


DATA_BATCH_STR = "data_batch_"
META_STR = "batches.meta"
CIFAR10 = "../../data/cifar-10-batches-py/"
DATA_ARRAY_STR = "data_array_batch_"


parser = argparse.ArgumentParser("Plot training images in CIFAR10 dataset.")
# parser.add_argument("-i", "--image", type=int, default=0, help="Index of the image in ")
parser.add_argument("-b", "--batch", type=int, default=1, help="Number of batch (1-5)")
args = parser.parse_args()


def unpickle(file):
    """
    It is a function to unpickle the batch file.
    :param file: data_batch_1-5
    :return:
    """
    with open(file, 'rb') as fo:
        file_dict = pickle.load(fo, encoding='bytes')
    return file_dict


def get_img_data(data, img_idx):
    """
    It is a function to reshape the image data.
    :param data: image data
    :param img_idx: index of image
    :return: data array with shape (32, 32, 3)
    """
    # Get the image data np.ndarray
    im = data[b'data'][img_idx, :]

    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    # 1-D arrays.shape = (N, ) ----> reshape to (1, N, 1)
    # 2-D arrays.shape = (M, N) ---> reshape to (M, N, 1)
    img = np.dstack((im_r, im_g, im_b))
    # img.shape = (32, 32, 3)

    return img


def get_img_label(data, img_idx):
    """
    It is a function to get image label
    :param data: image data
    :param img_idx: index of image
    :return: image label
    """
    label = data[b'labels'][img_idx]
    return label


def get_img_category(data, meta, img_idx):
    """
    It is a function to get image category
    :param data: image data
    :param meta: batch meta file
    :param img_idx: index of image
    :return: category
    """
    category = meta[b'label_names'][data[b'labels'][img_idx]]
    return category


def get_data_batch(batch_num):
    """
    It is a function to get cifar-19 data batch.
    :param batch_num: number of batch
    :return: data
    """
    pass


def list_to_array(img_list, label_list, category_list):
    """
    It is a function to transform <List> to <np.ndarray>
    :param img_list: image list
    :param label_list: label list
    :param category_list: category list
    :return: data array
    """
    assert len(img_list) == len(label_list) == len(category_list)
    data_array = np.array((img_list, label_list, category_list)).T
    return data_array


def save_data_array(filename, data_array):
    """
    It is a function to save data array
    :param filename: file name to save the data array
    :param data_array: Data array with (img_list, label_list, category_list)
    """
    np.save(filename, data_array)


def load_train_data_array(filename):
    """
    It is a function to load and combine data array.
    :param filename: name of the file saved with the data array
    :return: data_array
    """
    data_array = np.load(filename, allow_pickle=True)
    return data_array


def cifar10_plot(data, meta, img_idx=0):
    """
    It is a function to
    :param data: <dict> unpickle from data_batch_(1-5)
    :param meta:
    :param im_idx:
    :return:
    """
    img = get_img_data(data, img_idx)
    label = get_img_label(data, img_idx)
    category = get_img_category(data, meta, img_idx)

    print("img data.shape = ", img.shape)
    print("img data = ", img)
    print("img label = ", label)
    print("img category = ", category)
    print("max value in img data: ", np.max(img))

    # Plot image
    plt.imshow(img)
    plt.show()


def main():

    """
    ---------------------------
    Get every data batch
    ---------------------------
    """
    # Get batch number
    batch = args.batch
    print("It is now batch {} !".format(batch))

    # Unpickle data_batch_(1-5) and meta
    data = unpickle(os.path.join(CIFAR10, DATA_BATCH_STR + str(batch)))
    meta = unpickle(os.path.join(CIFAR10, META_STR))

    # Define <List> object
    batch_data_list = []
    batch_label_list = []
    batch_category_list = []

    for idx in range(len(data[b'data'])):

        # Get specific data
        img = get_img_data(data, img_idx=idx)
        label = get_img_label(data, img_idx=idx)
        category = get_img_category(data, meta, img_idx=idx)

        # Add to list
        batch_data_list.append(img)
        batch_label_list.append(label)
        batch_category_list.append(category)

    # Transform from <List> to <np.ndarray>
    data_array = list_to_array(batch_data_list, batch_label_list, batch_category_list)
    print("data_array.shape = ", data_array.shape)

    """
    ---------------------------
    Save data array as npy
    ---------------------------
    """
    if not os.path.exists(path="cifar10_batch_data"):
        os.mkdir("cifar10_batch_data")
    save_data_array(os.path.join("cifar10_batch_data", DATA_ARRAY_STR + str(batch)), data_array)
    print("Save Successfully!\n")

    """
    ---------------------------
    Plot image in CIFAR-10
    ---------------------------
    """
    # For plot
    # batch = (args.image // 10000) + 1
    # idx = args.image - (batch - 1) * 10000
    # cifar10_plot(data, meta, im_idx=idx)


if __name__ == "__main__":

    main()