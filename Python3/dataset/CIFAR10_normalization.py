# coding=utf8
"""
@author: Yantong Lai
@date: 09/16/2019
"""

import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset

from dataset.CIFAR10 import *

# np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', 10000)
pd.set_option('display.width',1000)


DATA_BATCH_STR = "data_batch_"
META_STR = "batches.meta"
CIFAR10 = "../../data/cifar-10-batches-py/"
DATA_ARRAY_STR = "data_array_batch_"
CIFAR10_BATCH_DIR = "cifar10_batch_data"


def normalization(batch_data_array):
    """
    It is a function to normalize data array
    :param batch_data_array: data array
    :return: normalized data_array
    """
    for idx in range(len(batch_data_array)):
        batch_data_array[idx, :, 0] /= 255
    return batch_data_array


def main():

    """
    ---------------------------
    Load data array as np.ndarray
    ---------------------------
    """
    batch_data_list = []
    for item in os.listdir(CIFAR10_BATCH_DIR):
        batch_data = load_train_data_array(os.path.join(CIFAR10_BATCH_DIR, item))
        batch_data_list.append(batch_data)

    """
    ---------------------------
    Save batch_data_array as npy
    ---------------------------
    """
    batch_data_array = np.array((batch_data_list))    # batch_data_array.shape = (5, 10000, 3)
    save_data_array(filename="all_batch_data", data_array=batch_data_array)
    print("Save batch_data_array Successfully!\n")

    """
    ---------------------------
    Normalize batch_data_array
    ---------------------------
    """
    norm_batch_data_array = normalization(batch_data_array)
    print("norm_batch_data_array.shape = ", norm_batch_data_array.shape)

    """
    ---------------------------
    Save norm_batch_data_array as npy
    ---------------------------
    """
    save_data_array(filename="norm_all_batch_data", data_array=norm_batch_data_array)
    print("Save norm_all_batch_data Successfully!\n")


def RandRecord(data, k):
    """
    It is a function to randomize k features
    :param data: <np.ndarray> data record
    :param k: number of features to modify
    :return: <np.ndarray> modified data record
    """
    if len(data.shape) == 2:
        if k == 0:
            # Initialize a data record
            data = np.random.uniform(low=0, high=1, size=data.shape[1])
        elif k < 0:
            raise ValueError("k < 0!")
        else:
            idx_to_modify = np.random.randint(low=0, high=data.shape[1], size=k)
            new_feats = np.random.uniform(low=0, high=1, size=k)
            data[0, idx_to_modify] = new_feats

        return data.reshape(1, -1)

    elif len(data.shape) == 3:
        if k == 0:
            # Initialize a data record
            data = np.random.uniform(low=0, high=1, size=(data.shape[0], data.shape[1]))
        elif k < 0:
            raise ValueError("k < 0!")
        else:
            pass
        return data



if __name__ == '__main__':

    main()