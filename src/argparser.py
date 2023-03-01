#
# System Arguments Parser
# By : Cameron Kaminski
#
# This file contains the function for parsing the system arguments using 'argparse'.
#

import argparse


def parseHyperparams():
    """
    Takes the system arguments and returns a dictionary of the hyperparameters.

    @return args : dict
        dictionary of the system arguments
    """
    parser = argparse.ArgumentParser(description='Hyperparameters for the Slow Feature Learning Network')
    parser.add_argument("--input_size", type=int, defautl=784, help="Input size of the network (default : 784)")
    parser.add_argument("--middle_width", type=int, default=2000, help="Width of the middle layer (default : 2000)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (default : 2)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--regular_lr", type=float, required=True, help="Regular learning rate")
    parser.add_argument("--slow_lr", type=float, required=True, help="Slow learning rate")
    parser.add_argument("--mnist_values", type=int, nargs='+', required=True, help="MNIST values to use")
    return parser.parse_args()
