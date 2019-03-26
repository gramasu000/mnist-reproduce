"""Module containing functions that create/load model checkpoint files"""

import pickle


def load_weights(filepath, neural_network):
    """Loads model parameters from file to neural network object 

    Unpickles a checkpoint file of network weights,
        and sets neural network to those weights

    Args:
        filepath (str): File path of pickled network weights
        neural_network (FourPointSevenNet): Neural Network as defined in net/net.py module
    """
    weights_data = pickle.load(filepath)
    neural_network.set_specific_weights(weights_data)


def save_weights(neural_network, filepath):
    """Saves model parameters from neural network object to file

    Pickles the weights of a neural network,
        and saves them in a checkpoint file

    Args:
        neural_network (FourPointSevenNet): Neural Network as defined in net/net.py module
        filepath (str): File path of pickled network weights
    """
    weights_data = neural_network.weights
    pickle.dump(weights_data, filepath)

