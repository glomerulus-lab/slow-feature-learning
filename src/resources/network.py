# Network Class
#
# Contains the NN class for the slow-feature-learning model.
#
# By Cameron Kaminski

import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from torch.nn.functional import one_hot

"""
Neural Network Class
"""
class NN(nn.Module):

    def __init__(self, middle_width, classes):

        super(NN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('hidden_layer', nn.Linear(784, middle_width)),
            ('hidden_activation', nn.ReLU()),
        ]))

        self.readout = nn.Linear(middle_width, classes)

        # Hyper Parameters

    # Forward pass.
    def forward(self, x):
        return self.readout(self.features(x))
