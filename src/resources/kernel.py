# C.K.A. Functions
#
# Descr.: Contains the functions for calculating the centered kernel analysis of a model, given the models features and phi.
#
# By Cameron G. Kaminski

import torch
def kernel_calc(y, phi):

    # Output Kernel
    y = torch.t(torch.unsqueeze(y, -1))
    K1 = torch.matmul(torch.t(y), y)
    K1c = kernel_centering(K1.float())

    # Feature Kernel
    K2 = torch.mm(phi, torch.t(phi))
    K2c = kernel_centering(K2)

    return kernel_alignment(K1c, K2c)


def frobenius_product(K1, K2):
    return torch.trace(torch.mm(K2, torch.t(K1)))


def kernel_alignment(K1, K2):
    return frobenius_product(K1, K2) / ((torch.norm(K1, p='fro') * torch.norm(K2, p='fro')))


def kernel_centering(K):
    # Lemmna 1

    m = K.size()[0]
    I = torch.eye(m)
    l = torch.ones(m, 1)

    # I - ll^T / m
    mat = I - torch.matmul(l, torch.t(l)) / m

    return torch.matmul(torch.matmul(mat, K), mat)