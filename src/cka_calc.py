#
# Center Kernel Alignment Calculator
# BY : CAMERON KAMINSKI
# DESCRIPTION : this script calculates the cka for all the models in the model
# save directory.
#


from resources import mnist_dataset
from resources import kernel_calc
from resources import get_model_saves
from resources import set_device
from resources import train
from resources import NN
import time
import argparse
import torch
import torch.nn as nn


def parseHyperparams() -> list[int]:
    """
    Takes the system argument and returns
    @return arg: mnist digit pair.
    """
    parser = argparse.ArgumentParser(description="CKA calculator")
    parser.add_argument("--mnist_values", type=int, nargs='+', required=True, help="MNIST values to use")
    return parser.parse_args()


device = set_device()

# Load MNIST dataset 
MNIST = mnist_dataset(batch_size=0, train=True, values=[0, 1])
digits = parseHyperparams()
data, targets = next(iter(MNIST))
data = torch.squeeze(data, dim=1)
data = data.view(data.size(0), -1)

model_paths = get_model_saves("model-saves")
num_models = len(model_paths)

losses = torch.zeros(num_models)
ckas = torch.zeros(num_models)

model = NN()
params = list(model.parameters())

i = 0
for model_path in model_paths:

    print(model_path)

    start_time = time.time()
    # load the model
    if model_path[-15:-13] != MNIST_values:
        MNIST_values = model_path[-15:-13]
        digits = [int(char) for char in model_path[-15:-13]]
        MNIST = mnist_dataset(batch_size=0, train=True, values=digits)
        print("GETTING NEW DATASET FOR DIGITS {digits}")
    end_time = time.time()
    print(f"TIME TO LOAD DATASET: {end_time - start_time:.2f}s")

    start_time = time.time()
    state_dict = torch.load(model_path, map_location=device)

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
    params[0].data = f_weights_float
    params[1].data = f_bias_float
    params[2].data = r_weights_float
    params[3].data = r_bias_float
    end_time = time.time()
    print(f"TIME TO LOAD MODEL: {end_time - start_time:.2f}s")

    # Getting CKA
    start_time = time.time()
    cka = kernel_calc(targets, model.features(data))
    end_time = time.time()
    print(f"TIME TO CALCULATE CKA: {end_time - start_time:.2f}s")

    # Getting Loss
    start_time = time.time()
    model.eval()
    loss = train(loader=MNIST, model=model, device=device, loss_function=nn.MSELoss(), values=digits, backwards=False,
                 record_loss=True)
    end_time = time.time()

    losses[i] = loss
    ckas[i] = cka

    print(f"Model {i} / {num_models} : Loss = {loss:.8f}, CKA = {cka:.8f}")
    i += 1
