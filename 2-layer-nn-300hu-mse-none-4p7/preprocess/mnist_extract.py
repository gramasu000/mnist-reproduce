import os
import numpy as np

from mnist_check import _MNIST_SIZES, _MNIST_NUM_EXAMPLES, 
                        _MNIST_DIMENSIONS, _MNIST_OFFSETS

class MnistExtractor:

    def __init__(self, image_filepath, label_filepath):
        self.fimage = open(image_filepath, "rb")
        self.flabel = open(label_filepath, "rb")

    def _extract_image_batch(self, start, end):
        np_array = np.empty((end-start, _MNIST_DIMENSIONS["image"]))
        self.fimage.seek(_MNIST_OFFSETS["image"] + start*_MNIST_DIMENSION["image"])
        for i in range(end-start):
            for j in range(num_feat):
                pxl_val = int.from_bytes(self.fimage.read(1), 
                                    byteorder="big", signed=False)
                np_array[i,j] = float(pxl_val)
        return np_array
                 
    def _extract_label_batch(self, start, end):
        np_array = np.zeros((end-start, _MNIST_DIMENSIONS["label"]))
        self.flabel.seek(_MNIST_OFFSETS["label"] + start)
        for i in range(end-start):
            label_int = int.from_bytes(self.flabel.read(1), 
                                    byteorder="big", signed=False)
            np_array[i, label_int] = 1
        return np_array

    def extract_batch(self, start, end):
        return _extract_image_batch(self, start, end),
                _extract_label_batch(self, start, end)

    def __del__(self):
        self.fi.close()
        self.fl.close()
