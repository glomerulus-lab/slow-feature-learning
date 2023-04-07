#
# Center Kernel Alignment Calculator
# BY : CAMERON KAMINSKI
# DESCRIPTION : this script calculates the cka for all the models in the model
# save directory.
#


from resources import kernel_calc
from resources import get_model_saves
from resources import 
import torch
import torch.nn as nn

# Load MNIST dataset 
#TODO-load MNIST such that we will only have to load the dataset 45 times

MNIST = null
data, targets = next(iter(MNIST))
data = torch.squeeze(data, dim=1)
data = data.view(data.size(0), -1)

model_paths = get_model_saves("model-saves")

for model_path in model_paths:
    

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # get the weights and biases of the quantized model (for the features layer)
    f_weights_quant = state_dict[
        'features.hidden_layer._packed_params._packed_params'][0]
    f_bias_quant = state_dict[
        'features.hidden_layer._packed_params._packed_params'][1]
    
    # dequantize the weights and biases
    f_weights_float = torch.dequantize(f_weights_quant)
    f_bias_float = torch.dequantize(f_bias_quant)

    # get the weights and biases of the quantized model (for the readout layer)
    r_weights_quant = state_dict['readout._packed_params._packed_params'][0]
    r_bias_quant = state_dict['readout._packed_params._packed_params'][1]
    
    # dequantize the weights and bises
    r_weights_float = torch.dequantize(r_weights_quant)
    r_bias_float = torch.dequantize(r_bias_quant)

    # Maunally update the model weights 
    model = NN()
    params[0].data = f_weights_float
    params[1].data = f_bias_float
    params[2].data = r_weights_float
    params[3].data = r_bias_float
    
    cka = kernel_calc(targets, data)
