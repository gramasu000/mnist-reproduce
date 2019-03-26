"""Module for input preprocessing operations, such as normalization"""

def fast_normalize(inputs):
    """Normalizes pixel values via division by 255

    Args:
        inputs (numpy.ndarray) : Flattened (C-style) image data (number_of_examples, 784)
    
    Returns:
        numpy.ndarray : The same array with each element divided by 255   
    """
    return inputs / 255
