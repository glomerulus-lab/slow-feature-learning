"""
LOAD MODEL INTO MEMORY
This module can dequantized models and loads them back into model object.
AUTHOR: CAMERON KAMINSKI
CREATED ON: 05.24.2023
LAST MODIFIED: 05.24.2023
"""

import torch.nn as nn
from model_utils import NN
import torch

def load_model(PATH: str, device: str) -> torch.nn.Module:
    """ Loads qunatised model from PATH and returns model object.
    :param PATH: str, path to model
    :param device: str, device to load model onto
    :return: torch.nn.Module, model object
    """

    state_dict = torch.load(PATH, map_location=device)

    # Unpacking the feature params 
    l1_weights_quant = state_dict['features.hidden_layer._packed_params.' \
                                  '_packed_params'][0]
    l2_bias_quant = state_dict['features.hidden_layer._packed_params.' \
                               '_packed_params'][1]
    # Dequantizing the feature params
    l1_weights = torch.dequantize(l1_weights_quant)
    l1_bias = torch.dequantize(l2_bias_quant)

    # Unpacking the readout layer
    l2_weights_quant = state_dict['readout._packed_params._packed_params'][0]
    l2_bias_quant = state_dict['readout._packed_params._packed_params'][1]
    # dequantize the weights and bises
    l2_weights= torch.dequantize(l2_weights_quant)
    l2_bias= torch.dequantize(l2_bias_quant)

    # Manually updating the model
    model = NN()
    params = list(model.parameters())
    params[0].data = l1_weights
    params[1].data = l1_bias
    params[2].data = l2_weights
    params[3].data = l2_bias

    return model

