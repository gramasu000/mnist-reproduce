"""module for miscellaneous image input operations

Includes functions for normalization, 
converting JPG/PNG images to 28x28 greyscale images and back. 
"""
from PIL import Image

_ROWS = 28
# Row dimension of MNIST image
_COLS = 28
# Column dimension of MNIST image

def fast_normalize(inputs):
    """Normalizes pixel values via division by 255

    Args:
        inputs (numpy.ndarray) : Flattened (C-style) image data (number_of_examples, 784)
    
    Returns:
        numpy.ndarray : The same array with each element divided by 255   
    """
    return inputs / 255

def convert_to_image(inputs, index=0):
    """Writes a numpy image to JPG image file

    Args:
        inputs (numpy.ndarray) : Flattened (C-style) image data (number_of_examples, 784)
        index (int) : Index of image to be written 
    """
    image_array = inputs[index, :]
    image_array = np.reshape(image_array, (_ROWS, _COLS))
    image = Image.fromarray(image_array, mode="L")
    image.save(f"assets/image_{index}.jpg", "JPEG")
