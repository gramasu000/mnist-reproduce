import numpy as np
import os

class FileChecker:
    def __init__(self):
        self._mnist_sizes = {
            "train_image": 47040016,
            "test_image": 7840016,
            "train_label": 60008,
            "test_label": 10008
        }
        self._mnist_magic_num = {
            "image": 2051,
            "label": 2049
        }
        self._mnist_num_examples = {
            "train": 60000,
            "test": 10000
        }

    def check_file(self, f):
        for use in ["train", "test"]:
            for type in ["image", "label"]:
                if self._is_mnist(f, use, type)
                   return (use, type)
        return (None, None) 

    def _is_mnist(self, f, use, type):
        if not self._correct_file_size(f, use, type):
            return False
        if not self._correct_magic_number(f, type):
            return False
        if not self._correct_num_examples(f, use):
            return False
        return True

    def _correct_file_size(self, f, use, type):
        correct_size = self._mnist_sizes[f"{use}_{type}"]
        f.seek(0, os.SEEK_END)
        measured_size = f.tell()
        f.seek(0, os.SEEK_SET)
        return measured_size is correct_size            

    def _correct_magic_number(self, f, type):
        correct_magic_num = self._mnist_magic_num[type]
        f.seek(0, os.SEEL_SET)
        file_magic_num = int.from_bytes(f.read(4))
        f.seek(0, os.SEEK_SET)
        return file_magic_num is correct_magic_num

    def _correct_num_examples(self, f, use):
        correct_num_examples = self._mnist_num_examples[use]
        f.seek(4, os.SEEL_SET)
        file_num_examples = int.from_bytes(f.read(4))
        f.seek(0, os.SEEK_SET)
        return file_num_examples is correct_num_examples


def pxl_read(f):
    

def convert_image_to_numpy(f):
    f.seek(4, os.SEEK_SET)
    num_examples = int.from_bytes(f.read(4))
    num_rows = int.from_bytes(f.read(4))
    num_cols = int.from_bytes(f.read(4))
    num_pxls = num_rows * num_cols
    nparray = np.zeroes(num_examples, num_pxls)
    

def extract_file(filepath):
    fileChecker = FileChecker()
    with open(filepath, "rb") as f:
        use, type = fileChecker.check_file(f)
        if use is not None:
            result = convert_to_numpy(f, type)

class ImageMaker:

def open_file(filepath):
    with open(filepath, "rb") as f:
        f_bytes = f.read()
    magic_number = int.from_bytes(f_bytes[0:4], "big"))
    num_examples = int.from_bytes(f_bytes[4:8], "big"))
    if magic_number is 2051:
        num_rows = int.from_bytes(f_bytes[8:12], "big"))
        num_cols = int.from_bytes(f_bytes[12:16], "big"))
        num_pxls = num_rows * num_cols
        input = np.array((num_examples, num_pxls))
        for (i, byte) in enumerate(f_bytes):
            pxl_val = int.from_bytes(byte)
            input[i / num_pxls][i % num_pxls] = pxl_val


def 

open_file("assets/train-images-idx3-ubyte")
