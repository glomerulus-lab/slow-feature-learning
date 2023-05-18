"""
MATRIX CENTERING UTILS
Module contains the functions used for centering matrix objects
AUTHOR: CAMERON KAMINSKI
CREATED: 05.14.2023
LAST MODIFIED: 05.14.2023
"""

import torch

def vector_centering(x: torch.Tensor) -> torch.Tensor:
    """ Performs centering on a PyTorch tensor
    :param x: an (nx1) tensor
    :return: the centered version of the input
    """
    return x - torch.mean(x)
    

def matrix_centering(x: torch.Tensor) -> torch.Tensor:
    """ Performs centering a multi-dim PyTorch tensor
    :param x: an (nxd) tensor
    :return: the centered version of the input
    """
    return x - x.mean(dim=1, keepdim=True) - x.mean(dim=0, keepdim=True) + x.mean()

