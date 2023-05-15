"""
CENTER KERNEL ALIGNMENT CALCULATION PROGRAM
This program calculates the center kernel alignment (CKA) between a model and 
the dataset that it was trained on. This program loads the all models (of a 
specified digit pair) and MNIST subset, then calculates the CKA between each
model and the dataset.
AUTHOR: CAMERON KAMINSKI
CREATED: 05.14.2023
LAST MODIFIED: 05.14.2023
"""

#import utils
import argparse
import os

def parse_parameters():
    """ Takes the system arguments and return a dictionary of hyperparameters
    :return: dictionary of hyperparameters
    """
    parser = argparse.ArgumentParser(description='CKA Calculator')
    parser.add_argument('--mnistDigits', type=str, required=True, 
                        help='MNIST digit pair to use for training and testing')
    return parser.parse_args()

def get_model_paths(digits: str) -> str:
    """ Loads the model from the specified path
    :param PATH: path to the model
    :return: model paths
    """
    model_dir = os.path.join(os.getcwd(), "../model-saves")
    all_dirs = os.listdir(model_dir)
    target_dirs = [d for d in all_dirs if d.endswith(digits)]
    model_paths = []
    for target_dir in target_dirs:
        target_path = os.path.join(model_dir, target_dir)
        model_files = os.listdir(target_path)
        for model_file in model_files:
            model_path = os.path.join(target_path, model_file)
            model_paths.append(model_path)
    return model_paths

if __name__ == "__main__":
    args = parse_parameters()
    model_dir = get_model_paths(args.mnistDigits)
    print(model_dir)

