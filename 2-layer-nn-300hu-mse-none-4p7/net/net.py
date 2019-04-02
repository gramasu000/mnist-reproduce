"""Main module for Neural Networks

This module defines a simple neural network which achieved 4.7% error
for the MNIST dataset. Therefore, we call it the FourPointSevenNet.
"""
import numpy as np

from utils.log import LOG

class FourPointSevenNet:
    """A two-layer neural network (to be trained on MNIST)

    This network has an input layer of dimension 784,
        a hidden layer of dimension 300 and an output layer of dimension 10.
    In the hidden layer, we use ReLU activation,
        while in the output, we use softmax activation.
    """

    def __init__(self):
        """Sets the layer dimensions"""
        LOG.info("Initialize FourPointSevenNet - 784 x 300 x 10 dense network")
        self.input_size = 784
        self.hidden_size = 300
        self.output_size = 10

    def set_random_weights(self):
        """Sets weight and bias matrices to random normal values

        We call this method when training a network from scratch
        """
        LOG.info("Initialize weights of FourPointSevenNet - Random Initialization")
        w1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        b1 = np.random.randn(1, self.hidden_size) * 0.1
        w2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        b2 = np.random.randn(1, self.output_size) * 0.1
        self.weights = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2
        }

    def set_specific_weights(self, weights):
        """Sets weights and bias matrices according to input

        We call this method when we seek to set weights according to a
            pretrained model, for prediction or for transfer learning.

        Args:
            weights (dict) : A dictionary of numpy arrays with keys "w1", "b1", "w2", "b2"
        """
        LOG.info("Initialize weights of FourPointSevenNet - From Checkpoint")
        self.weights = weights

    def learning_rate(self, epoch):
        """Returns the learning rate for the epoch of training

        Args:
            epoch (int) - How many times the training cycle has been iterated

        Returns:
            float - learning rate for gradient descent

        Todo:
            Modify this to have a learning rate that depends on epoch number.
            For example, the learning rate can be reduced if the epoch number is high,
                which means and model is close to minimal loss
        """
        return 0.001

    def _forward_prop(self, inputs, labels):
        """Computes forward propogration

        Pass the MNIST input and labels through the network,
            and compute the loss value as well as intermediary results

        Args:
            inputs (numpy.ndarray) - Flattened (C-style) image data (number_of_examples, 784)
            labels (numpy.ndarray) - One-hot encoded label data (number_of_examples, 10) [or scalar 0]
        """
        LOG.debug(f"FourPointSevenNet - Forward Propagation")
        h1 = np.matmul(inputs, self.weights["w1"]) + self.weights["b1"]
        a1 = np.greater(h1, 0) * h1
        h2 = np.matmul(a1, self.weights["w2"]) + self.weights["b2"]
        a2 = np.exp(h2)
        a2 = a2 / np.sum(a2, axis=1)[:, None]
        loss = np.sum(-np.log(a2) * labels)
        LOG.info(f"FourPointSevenNet - Loss: {loss}")
        self.fprop = {
            "h1": h1,
            "a1": a1,
            "h2": h2,
            "a2": a2,
            "loss": loss
        }

    def predict(self, inputs):
        """Predict classification for input images

        Pass the input images through the neural network
        forward propogration, and return a vector of the labels
        with the highest probability.

        Args:
            inputs (numpy.ndarray) - Flattened (C-style) image data (number_of_examples, 784)

        Returns:
            preds (numpy.ndarray) - A vector of predictions [0-9] (number_of_examples, 1)
        """
        LOG.debug("FourPointSevenNet - Prediction")
        h1 = np.matmul(inputs, self.weights["w1"]) + self.weights["b1"]
        a1 = np.greater(h1, 0) * h1
        h2 = np.matmul(a1, self.weights["w2"]) + self.weights["b2"]
        a2 = np.exp(h2)
        a2 = a2 / np.sum(a2, axis=1)[:, None]
        preds = np.argmax(a2, axis=1)
        return preds

    def _compute_gradients(self, inputs, labels):
        """Computes the gradients (backpropogation) for all weight and bias matrices

        Uses the loss value, the inputs and labels,
        and the saved intermediary results to run backpropagation
        and compute the gradients for weights and bias matrices

        Args:
            inputs (numpy.ndarray) - Flattened (C-style) image data (number_of_examples, 784)
            labels (numpy.ndarray) - One-hot encoded label data (number_of_examples, 10) [or scalar 0]
        """
        LOG.debug("FourPointSevenNet - BackPropogation")
        dh2 = self.fprop["a2"] - labels
        dw2 = np.matmul(self.fprop["a1"].T, dh2)
        db2 = np.sum(dh2, axis=0)
        da1 = np.matmul(dh2, self.weights["w2"].T)
        dh1 = da1 * np.greater(self.fprop["h1"], 0)
        dw1 = np.matmul(inputs.T, dh1)
        db1 = np.sum(dh1, axis=0)
        self.bprop = {
            "dw2": dw2,
            "db2": db2,
            "dw1": dw1,
            "db1": db1
        }

    def _eval_grad_descent(self, epoch):
        """Computes gradient descent step

        Updates the weights and bias matrices
            according to the gradients and learning rate

        Args:
            epoch (int) - How many times the training cycle has been iterated
        """
        LOG.debug("FourPointSevenNet - Gradient Descent")
        for key in self.weights:
            bkey = "d" + key
            self.weights[key] -= self.learning_rate(epoch)*self.bprop[bkey]

    def train(self, epoch, inputs, labels):
        """Runs a full training interation

        Runs forward propogation, back propagation, and a gradient descent step

        Args:
            epoch (int) - How many times the training cycle has been iterated
            inputs (numpy.ndarray) - Flattened (C-style) image data (number_of_examples, 784)
            labels (numpy.ndarray) - One-hot encoded label data (number_of_examples, 10) [or scalar 0]
        """
        LOG.info(f"FourPointSevenNet - Training Iteration {epoch}")
        self._forward_prop(inputs, labels)
        self._compute_gradients(inputs, labels)
        self._eval_grad_descent(epoch)
