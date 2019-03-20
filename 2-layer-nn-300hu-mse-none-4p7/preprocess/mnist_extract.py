import os
import numpy as np

from mnist_check import check_file 

def _extract_images(f):
    f.seek(4, os.SEEK_SET)
    num_examples = int.from_bytes(f.read(4))
    num_rows = int.from_bytes(f.read(4))
    num_cols = int.from_bytes(f.read(4))
    num_feat = num_rows * num_cols
    np_array = np.empty((num_examples, num_feat))    
    for i in num_examples:
        for j in num_feat:
            pxl_val = int.from_bytes(f.read(1), byteorder="big", signed=False)
            np_array[i,j] = float(pxl_val)
    return np_array

def _extract_labels(f):
    f.seek(4, os.SEEK_SET)
    num_examples = int.from_bytes(f.read(4))
    np_array = np.zeros((num_examples, 10))    
    for i in num_examples:
        label_int = int.from_bytes(f.read(1), byteorder="big", signed=False)
        np_array[i, label_int] = 1
    return np_array

def extract(file_path):
    with open(file_path, "rb") as f:
        use, type = check_file(f)
        if type is "image":
            return _extract_images(f)
        if type is "label":
            return _extract_labels(f)
