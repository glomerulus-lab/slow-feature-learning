"""
MODEL UTILS
Module contating the underlying neural network model architecture
AUTHOR: CAMERON KAMINSKI
CREATED: 05.14.2023
LAST MODIFIED: 05.14.2023
"""

import torch
import torch.nn as nn   # Nueral network modules.
from collections import OrderedDict

class NN(nn.Module):

    def __init__(self, input_size: int=784, 
                 middle_width: int=2048, 
                 num_classes: int=2):


        super(NN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('hidden_layer', nn.Linear(input_size, middle_width)),
            ('hidden_activation', nn.ReLU()),
        ]))
        self.readout = nn.Linear(middle_width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.readout(x)

        return x

