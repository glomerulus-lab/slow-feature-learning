"""
CENTERED KERNEL ALIGNMENT UTILS
Moduel contains functions for calculating centered kernel alignment.
AUTHOR: CAMERON G. KAMINSKI
CREATED: 05.14.2023
LAST MODIFIED: 05.14.2023
"""

import torch


def kernel_alignment(y: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """ Calculates the kernel alignment between y (n x 1) and phi (n x d) 
    :param y: torch.Tensor, a target vector
    :param phi: torch.Tensor, the model features matrix
    :return: kernel alignment of the target vecotr and the model featre matrix
    """
    v = phi.T @ y
    return v.T @ v / (torch.norm(y)**2 * torhc.norm(phi.T @ phi))

