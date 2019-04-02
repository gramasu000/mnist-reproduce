import os
import logging

from utils.log import init_logger
init_logger("logs/train.log", logging.INFO)
from utils.log import LOG

from net.net import FourPointSevenNet
from mnist.mnist_info import MNIST_FILENAMES, MNIST_NUM_EXAMPLES
from mnist.mnist_check import check_mnist  
from mnist.mnist_extract import MnistExtractor 

from utils.batch_generate import BatchIndicesGenerator
from utils.preprocess import fast_normalize 

if __name__ == "__main__":
    check_mnist()
    net = FourPointSevenNet()
    b_gen = BatchIndicesGenerator(100, MNIST_NUM_EXAMPLES["train"])
    tr_ext = MnistExtractor(MNIST_FILENAMES["train_image"], MNIST_FILENAMES["train_label"])
    net.set_random_weights()
    for epoch in range(100):
        batch_indices = b_gen.gen_batch()
        inputs, labels = tr_ext.extract_batch(batch_indices)
        inputs = fast_normalize(inputs)
        net.train(epoch, inputs, labels)
    tr_ext.close()
