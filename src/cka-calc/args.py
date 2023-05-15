"""
COMMAND LINE ARGUMENTS FOR CKA-CALC.PY
"""

import argparse

def parse_hyperparameters():
    """ Takes the system arguments and return a dictionary of hyperparameters
    :return: dictionary of hyperparameters
    """
    parser = argparse.ArgumentParser(description='CKA Calculator')
    parser.add_argument('--mnistDigits', type=str, required=True, 
                        help='MNIST digit pair to use for training and testing')
    return parser.parse_args()


