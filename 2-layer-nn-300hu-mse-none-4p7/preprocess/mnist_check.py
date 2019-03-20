import os

_MNIST_SIZES = {
    "train_image": 47040016,
    "test_image": 7840016,
    "train_label": 60008,
    "test_label": 10008
}

_MNIST_MAGIC_NUM = {
    "image": 2051,
    "label": 2049
}

_MNIST_NUM_EXAMPLES = {
    "train": 60000,
    "test": 10000
}

DIR = "assets"

MNIST_FILENAMES = {
    "train_image": "train-images-idx3-ubyte",
    "test_image": "t10k-images-idx3-ubyte",
    "train_label": "train-labels-idx1-ubyte",
    "test_label": "t10k-labels-idx1-ubyte"
}

def _correct_size(f, use, type):
    correct_fsize = _MNIST_SIZES[f"{use}_{type}"]
    f.seek(0, os.SEEK_END)
    fsize = f.tell()
    f.seek(0, os.SEEK_SET)
    return fsize is correct_fsize

def _correct_magic_num(f, type):
    correct_fmagicnum = _MNIST_MAGIC_NUM[type]
    fmagicnum = int.from_bytes(f.read(4), "big")
    f.seek(0, os.SEEK_SET)
    return fmagicnum is correct_fmagicnum

def _correct_num_examples(f, use):
    correct_fnumexamples = _MNIST_NUM_EXAMPLES[use]
    f.seek(4, os.SEEK_SET)
    fnumexamples = int.from_bytes(f.read(4), "big")
    f.seek(0, os.SEEK_SET)
    return fnumexamples is correct_fnumexamples

def _is_mnist(f, use, type):
    return _correct_size(f, use, type) 
        and _correct_magic_num(f, type) 
        and _correct_num_examples(f, use)

def check_file(f):
    for use in ["train", "test"]:
        for type in ["image", "label"]:
            if _is_mnist(f, use, type):
                return use, type
    return None, None
