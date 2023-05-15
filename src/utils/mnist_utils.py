"""
MINIST UTILS
Module containing functions for loading and preprocessing the MNIST dataset.
AUTHOR: CAMERON G. KAMINSKI
CREATED: 05.14.2023
LAST MODIFIED: 05.14.2023
"""

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def mnist_dataset(batch_size: int, 
                  train: bool=True,
                  values: [int]=list(range(10))):
    """Loads mnist dataset into a dataloader object.
    :param batch_size: batch size for dataloader object.
    :param train: boolean indicating whether to load training or test data.
    :param values: list of integers indicating which MNIST targets to load.
    :return: dataloader object containing MNIST data.
    """

    # Initializing MNIST data set.
    dataset = datasets.MNIST(root='dataset/',
                             train=train,
                             transform=transforms.ToTensor(),
                             download=False)

    targets_list = dataset.targets.tolist()
    values_index = [i for i in range(len(dataset)) if targets_list[i] in values]

    # Creating a subset of ### MNIST targets.
    subset = torch.utils.data.Subset(dataset, values_index)
    loader = DataLoader(dataset=subset, shuffle=True)

    return loader
