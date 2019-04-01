"""Module to extract MNIST data from binary files

Defines a class MnistExtractor which takes in filepaths
for the image and label binaries, and then extracts 
batch data in numpy form 
"""
import os
import numpy as np

from .mnist_info import MNIST_SIZES, MNIST_NUM_EXAMPLES,\
                         MNIST_DIMENSIONS, MNIST_OFFSETS
from ..utils.log import LOG

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
        LOG.info(f"Initialized MnistExtractor Object - Opened files {image_filepath} and {label_filepath}")
        self.fimage = open(image_filepath, "rb")
        self.flabel = open(label_filepath, "rb")

    def _extract_image_batch(self, list_of_indices):
        """Read batch image data into numpy array (flattened)

        Args:
            list_of_indices (list): List of indices of examples in batch
        
        Returns:
            numpy.ndarray: Flattened (C-style) array of batch image data - shape: (len(list_of_indices), 784)
        """ 
        batch_size = len(list_of_indices)
        np_array = np.empty((batch_size, MNIST_DIMENSIONS["image"]))
        for i in list_of_indices:
            self.fimage.seek(MNIST_OFFSETS["image"] + i*MNIST_DIMENSION["image"], os.SEEK_SET)
            for j in range(MNIST_DIMENSIONS["image"]):
                pxl_val = int.from_bytes(self.fimage.read(1), byteorder="big", signed=False)
                np_array[i,j] = float(pxl_val)
        LOG.debug(f"Extracting flattened image batch - {list_of_indices}")
        return np_array
                 
    def _extract_label_batch(self, list_of_indices):
        """Read batch label data to numpy array (one-hot encoded)

        Args:
            list_of_indices (list): List of indices of examples in batch
            
        Returns:
            numpy.ndarray: One-hot encoded array of batch label data - shape: (len(list_of_indices), 10)
        """
        batch_size = len(list_of_indices)
        np_array = np.zeros((batch_size, MNIST_DIMENSIONS["label"]))
        for i in list_of_indices:
            self.flabel.seek(MNIST_OFFSETS["label"] + i, os.SEEK_SET)
            label_int = int.from_bytes(self.flabel.read(1), byteorder="big", signed=False)
            np_array[i, label_int] = 1
        LOG.debug(f"Extracting one-hot-encoded label batch ({start}, {end})")
        return np_array

    def extract_batch(self, list_of_indices):
        """Read and return batch data - (images, labels)
        
        Args:
            list_of_indices (list): List of indices of examples in batch
        
        Returns:
            numpy.ndarray, numpy.ndarray : batch image data, batch label data  
        """
        LOG.info(f"Extracting a image/label batch ({start}, {end})")
        return _extract_image_batch(self, list_of_indices), _extract_label_batch(self, list_of_indices)

    def __del__(self):
        """Closes file objects"""
        LOG.info("Destroying MnistExtractor - Closing file objects for {self.fi.name} and {self.fl.name}")
        self.fi.close()
        self.fl.close()
