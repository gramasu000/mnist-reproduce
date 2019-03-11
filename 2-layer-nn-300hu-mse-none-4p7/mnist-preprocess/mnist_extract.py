import os
import numpy as np

from mnist_check import check_file 

DIRECTORY = "assets"
FILES = ["train-images-idx3-ubyte", "train-labels-idx3-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx3-ubyte"]

def _extract_images(f):
    f.seek(4, os.SEEK_SET)
    num_examples = int.from_bytes(f.read(4))
    num_rows = int.from_bytes(f.read(4))
    num_cols = int.from_bytes(f.read(4))
    num_feat = num_rows * num_cols
    np_array = np.empty((num_examples, num_feat))    
    for i in num_examples:
        for j in num_feat:
            np_array[i,j] = float(int.from_bytes(f.read(1)))
    return np_array

def _extract_labels(f):
    f.seek(4, os.SEEK_SET)
    num_examples = int.from_bytes(f.read(4))
    np_array = np.empty((num_examples,))    
    for i in num_examples:
        np_array[i] = float(int.from_bytes(f.read(1)))
    return np_array

def extract_all():
    np_arrays = []
    for file in FILES:
        file_path = DIRECTORY + os.sep + file
        with open(file_path, "rb") as f:
            use, type = check_file(f)
            if type is "image":
                np_arrays.append(_extract_images(f))
            if type is "label":
                np_arrays.append(_extract_labels(f))
    return np_arrays

