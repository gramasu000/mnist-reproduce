"""Module to extract MNIST data from binary files

Defines a class MnistExtractor which takes in filepaths
for the image and label binaries, and then extracts 
batch data in numpy form 
"""
import os
import numpy as np

from mnist_check import _MNIST_SIZES, _MNIST_NUM_EXAMPLES, _MNIST_DIMENSIONS, _MNIST_OFFSETS

class MnistExtractor:
    """A class to extract MNIST data
    
    An MnistExtractor object can be initialized
    with the filepaths of the images and labels.
    Then, we call the extract_batch method to output
    any batch of data in numpy array form.
    """

    def __init__(self, image_filepath, label_filepath):
        """Open file objects for MNIST images and labels 

        Args:
            image_filepath (str) : File path to MNIST image data
            label_filepath (str) : File path to MNIST label data

        Note:
            number_of_examples must be same for both 
            image_filepath and label_filepath
        """
        self.fimage = open(image_filepath, "rb")
        self.flabel = open(label_filepath, "rb")

    def _extract_image_batch(self, start, end):
        """Read batch image data into numpy array (flattened)

        Args:
            start (int) : Index of first example in batch
            end (int) : Index of last example in batch + 1 (first example not in batch)
        
        Returns:
            numpy.ndarray : Flattened (C-style) array of batch image data - shape: (end-start, 784)
        """ 
        np_array = np.empty((end-start, _MNIST_DIMENSIONS["image"]))
        self.fimage.seek(_MNIST_OFFSETS["image"] + start * _MNIST_DIMENSION["image"])
        for i in range(end-start):
            for j in range(num_feat):
                pxl_val = int.from_bytes(self.fimage.read(1), 
                                    byteorder="big", signed=False)
                np_array[i,j] = float(pxl_val)
        return np_array
                 
    def _extract_label_batch(self, start, end):
        """Read batch label data to numpy array (one-hot encoded)

        Args:
            start (int) : Index of first example in batch
            end (int) : Index of last example in batch + 1 (first example not in batch)
            
        Returns:
            numpy.ndarray : One-hot encoded array of batch label data - shape: (end-start, 10)
        """ 
        np_array = np.zeros((end-start, _MNIST_DIMENSIONS["label"]))
        self.flabel.seek(_MNIST_OFFSETS["label"] + start)
        for i in range(end-start):
            label_int = int.from_bytes(self.flabel.read(1), 
                                    byteorder="big", signed=False)
            np_array[i, label_int] = 1
        return np_array

    def extract_batch(self, start, end):
        """Read and return batch data - (images, labels)
        
        Args:
            start (int) : Index of first example in batch
            end (int) : Index of last example in batch + 1 (first example not in batch)
        
        Returns:
            numpy.ndarray, numpy.ndarray : batch image data, batch label data  
        """
        return _extract_image_batch(self, start, end), _extract_label_batch(self, start, end)

    def __del__(self):
        """Closes file objects"""
        self.fi.close()
        self.fl.close()
