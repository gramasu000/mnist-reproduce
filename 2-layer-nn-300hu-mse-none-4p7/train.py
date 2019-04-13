import os
import numpy
import logging
import argparse

from utils.log import init_logger
init_logger("logs/train.log", logging.INFO)
from utils.log import LOG

from net.net import FourPointSevenNet
from mnist.mnist_info import MNIST_FILENAMES, MNIST_NUM_EXAMPLES
from mnist.mnist_check import check_mnist  
from mnist.mnist_extract import MnistExtractor 

from utils.batch_generate import BatchIndicesGenerator
from utils.preprocess import fast_normalize 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--batchsize", type=int, help="batch size used for training")
    parser.add_argument("-v", "--validation", type=int, help="size of validation dataset")
    parser.add_argument("-n", "--numepochs", type=int, help="number of epochs to train neural network")
    parser.add_argument("-e", "--earlystopping", action="store_true", help="save weights according to early stopping") 
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments() 
    check_mnist()
    
    net = FourPointSevenNet()
    gen = BatchIndicesGenerator(args.batchsize, MNIST_NUM_EXAMPLES["train"] - args.validation, MNIST_NUM_EXAMPLES["train"])
    ext = MnistExtractor(MNIST_FILENAMES["train_image"], MNIST_FILENAMES["train_label"])
    
    train_losses = []
    val_losses = []
    min_val_loss = numpy.Inf

    net.set_random_weights()


    for epoch in range(args.numepochs):
    
        batch_indices = gen.gen_train_batch()
        batch_inputs, batch_labels = ext.extract_batch(batch_indices)
        batch_inputs = fast_normalize(batch_inputs)
    
        net.train(epoch, inputs, labels)
        
        # Record Results
        train_indices = gen.gen_train()
        train_inputs, train_labels = ext.entract_batch(batch_indices)
        train_inputs = fast_normalize(train_inputs)
        train_losses.append(net.loss(train_inputs, train_labels))
        
        val_indices = gen.gen_validate()
        val_inputs, val_labels = ext.extract_batch(batch_indices)
        val_inputs = fast_normalize(val_inputs)
        val_losses.append(net.loss(val_inputs, val_labels))

        if val_losses[-1] < min_val_loss:
            min_val_loss = val_losses[-1]
            # Check to see if ckpt file exists, and delete it
            save_weights(net, "output/early-stopping.ckpt") 
    
    # TODO: Save the losses as text files
    # TODO: Use matplotlib to plot the losses
    # TODO: Save the weights again 

    tr_ext.close()
