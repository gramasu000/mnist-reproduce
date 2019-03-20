import os

from net.net import FourPointSevenNet
from preprocess.mnist_check import MNIST_FILENAMES, MNIST_NUM_EXAMPLES, check_mnist  
from preprocess.mnist_extract import MnistExtractor 


class BatchIndicesGenerator():
    
    def __init__(self, batch_size, num_examples):
        self.index_tuples = []
        index = batch_size
        while index < num_examples:
            self.index_tuples.append((index-batch_size, index))
            index += batch_size
        self.index_tuples.append((index-batch_size, num_examples))
        self.size = len(self.index_tuples)
    
    def gen_batch(self, epoch):
        return self.index_tuples[epoch % self.size]


if __name__ == "__main__":
    net = FourPointSevenNet()
    b_gen = BatchIndicesGenerator(100, _MNIST_NUM_EXAMPLES["train"])    
    tr_ext = MnistExtractor(MNIST_FILENAMES["train_image"],
                                        MNIST_FILENAMES["train_label"])
    for epoch in range(400):
        start, end = b_gen.gen_batch(epoch) 
        inputs, labels = tr_ext.extract_batch(start, end)
        net.train(epoch, inputs, labels)
