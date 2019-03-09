import numpy as np

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

open_file("assets/train-images-idx3-ubyte")
