# MNIST Data loader functions 
# 
# Descr.: Contains the functions for loading the mnist dataset & it's subsets.
# 
# By Cameron G. Kaminski

import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

def mnist_dataset(batch_size, train=True, values=list(range(10))):
    # Initializing MNIST data set.
    dataset = datasets.MNIST(root='dataset/', train=train, transform=transforms.ToTensor(), download=False)

    targets_list = dataset.targets.tolist()
    values_index = [i for i in range(len(dataset)) if targets_list[i] in values]

    # Creating a subset of ### MNIST targets.
    subset = torch.utils.data.Subset(dataset, values_index)
    loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)

    return loader