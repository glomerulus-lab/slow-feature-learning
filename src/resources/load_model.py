# 
# LOAD PYTORCH MODEL 
# BY : CAMERON KAMINSKI
# 

import torch
import os
from .network import NN


def load_model(lr: float, slr: float, digits: (int, int), epoch: int) -> NN:
    """
    Takes model specifications (i.e. learning rates and digit pair), then loads
    the pytorch model for the specified epoch.

    :param lr: learning rate
    :param slr: slow learning rate
    :param digits: digits used on model
    :param epoch: epoch that model was saved on
    :return: PyTorch model for the specified model
    :rtype: torch.nn
    """

    path = ("/model-saves" +
            str(lr) + '_' + str(slr) + '_' + str(digits(0)) + str(digits(1)) +
            "{:04d}".format(epoch))

    torch.load(path)
    model = NN(input_size=784, middle_width=2048, num_classes=2)

    return model


def get_model_saves(directory: str, mnist_digits: list[int]) -> list[str]:
    """
    Returns a list of all the model saves in the specified directory accessible from the current program's directory.

    :param directory: root directory to search for model saves
    :return paths: list of paths to model saves
    """
    paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("hyperparams.txt"):
                for model in os.listdir(root):
                    if file.endswith('.pt'):
                        paths.append(os.path.join(root, model))
    return paths
