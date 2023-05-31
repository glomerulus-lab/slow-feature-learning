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

import argparse
import csv
from typing import List, Tuple
import re
import torch
import os

# importing src utils
import sys
sys.path.append('../')
import utils


def parse_parameters():
    """ Takes the system arguments and return a dictionary of hyperparameters
    :return: dictionary of hyperparameters
    """
    parser = argparse.ArgumentParser(description='CKA Calculator')
    parser.add_argument('--mnistDigits', type=str, required=True, 
                        help='MNIST digit pair to use for training and testing')
    return parser.parse_args()

def get_model_paths(digits: str) -> list[tuple[str, list[str]]]:
    """ Loads the model from the specified path
    :param PATH: path to the model
    :return: model paths
    """
    model_dir = os.path.join(os.getcwd(), "../model-saves")
    all_dirs = os.listdir(model_dir)
    target_dirs = [d for d in all_dirs if d.endswith(digits)]
    data = []
    model_paths = []
    for target_dir in target_dirs:
        target_path = os.path.join(model_dir, target_dir)
        model_files = os.listdir(target_path)
        for model_file in model_files:
            if model_file.endswith(".pt"):
                model_path = os.path.join(target_path, model_file)
                model_paths.append((model_path,
                    target_dir.split("_") + re.findall(r'\d+', model_file)))
    model_paths_sorted = sorted(model_paths, key=lambda x: x[1][0])
    return model_paths_sorted


if __name__ == "__main__":

    # Hyper params
    args = parse_parameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the MNIST dataset for the digit pairs
    values = [int(char) for char in args.mnistDigits]
    mnist = utils.load_mnist_dataset(batch_size=0, values=[int(char) for char in args.mnistDigits])

    # Extracting the targets & data
    data, targets = next(iter(mnist))
    data = torch.squeeze(data, dim=1)
    data = data.view(data.size(0), -1)

    # Preparing the targets (y) vector for cka calculation
    y = torch.unsqueeze(targets.T, -1).float().to(device)
    y_c = utils.vector_centering(y)
    y_c_norm = torch.norm(y_c)

    # Loading all model paths (for the specified digit pair)
    model_dir = get_model_paths(args.mnistDigits)
    #model_dir.sort()

    # Calculating the cka for each model
    file_name = f"cka_results_{args.mnistDigits}.csv"
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["input_lr", "readout_lr", "digit_pair", "epoch", "CKA"])
        t = 0
        for model_path, hyper_params in model_dir:
            # Loading the model
            model = utils.load_model(model_path, device)

            # Prepare the model features' matrix (phi) for cka calculation
            phi = model.features(data)
            phi_c = utils.matrix_centering(phi)

            # Calculating the cka
            v = phi_c.T @ y_c
            cka = (v.T @ v) / (y_c_norm ** 2 * torch.norm(phi_c.T @ phi_c))
            print(f"CKA {model_path}: {cka.item()}")
            writer.writerow([hyper_params[0], hyper_params[1], hyper_params[2], hyper_params[3], cka.item()])
            t += 1
            if t == 10:
                break
