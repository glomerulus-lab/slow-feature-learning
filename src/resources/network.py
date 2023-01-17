import torch.nn as nn   # Nueral network modules.
from collections import OrderedDict

class NN(nn.Module):
    def __init__(self, middle_width, classes):
        super(NN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('hidden_layer', nn.Linear(784, middle_width)),
            ('hidden_activation', nn.ReLU()),
        ]))

        self.readout = nn.Linear(middle_width, classes)

    def forward(self, x):
        return self.readout(self.features(x))