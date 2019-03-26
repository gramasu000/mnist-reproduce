"""Module for mnist metadata"""

import os


MNIST_SHA3_512 = {
    "train_image": "4099cec88c23c5b662ddde6f13557bfaad0d9f2fe87461fe44c974d13c6030a008d40fec6bdf9934b9462d3c39644a7acfbaa62782bb2c9eead144782cd66708",
    "test_image": "087a6af735326ad7dd071e3d2878b6c4daef10d65d46c14ae1e5c8ab8c0d3f0a5898f9c702adcf470d6f8a09fce585bc9c96a66b83f2958736055db1185a867b",
    "train_label": "00f1a55e6be3a37e2e87b39c630b3a2da9be8146c4014571a0345a18ff9e7a33380b8bfecf5b9b225befe3873eb2312e4ec0c774691e63751ccc2b891358f5d8",
    "test_label": "af7dd0dc1ac098fefbcdb5e83648f53042d8a3f8eb806a38e06e3baed09aaccf247331de2f9e74cbafbdc4ccafa229e52d08968d1a17d683ef75ad0d385b663c"
}
"""dict: Contains the SHA3-512 hashes of the MNIST binary files"""


MNIST_SIZES = {
    "train_image": 47040016,
    "test_image": 7840016,
    "train_label": 60008,
    "test_label": 10008
}
"""dict: Contains the file size (in bytes) of the MNIST binary files"""


MNIST_MAGIC_NUM = {
    "image": 2051,
    "label": 2049
}
"""dict: Contains the magic numbers of the MNIST binary files

The magic number is the value of the first 4 bytes of the MNIST file, using big endian
"""


MNIST_NUM_EXAMPLES = {
    "train": 60000,
    "test": 10000
}
"""dict: Contains the number of dataset items for the MNIST binary files"""


MNIST_DIMENSIONS = {
    "image": 784,
    "label": 10
}
"""dict: Contains the dimensions of the MNIST data"""


MNIST_OFFSETS = {
    "image": 16,
    "label": 8
}
"""dict: Contains offsets for MNIST data

The offset shows how many bytes of the MNIST binary (from the beginning of the file)
contain metadata and can be skipped
"""


DIR = "assets"
"""str: directory containing the MNIST binary files"""


MNIST_FILENAMES = {
    "train_image": DIR + os.sep + "train-images-idx3-ubyte",
    "test_image": DIR + os.sep + "t10k-images-idx3-ubyte",
    "train_label": DIR + os.sep + "train-labels-idx1-ubyte",
    "test_label": DIR + os.sep + "t10k-labels-idx1-ubyte"
}
"""dict: Contains filepaths to MNIST binary files"""
