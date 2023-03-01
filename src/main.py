from network_functions import *
import os
from numba import jit
from argparser import parseHyperparams
from network import *
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import pandas as pd

if __name__ == '__main__':
    # Checking & Setting Device Allocation
    device = set_device()
    print(f"Running on {device}")

    # Parsing Hyperparameters
    hyper_params = parseHyperparams()
    print(f"Hyper Parameters: {hyper_params}")

    # Initializing Model
    model = NN(input_size=hyper_params.input_size,
               middle_width=hyper_params.middle_width,
               num_classes=hyper_params.num_classes).to(device=device)

    # Loading MNIST Dataset
    print(f"MNIST digits {hyper_params.mnist_values}")
    train_loader = mnist_dataset(hyper_params.batch_size, values=hyper_params.mnist_values)
    validate_loader = mnist_dataset(hyper_params.batch_size, train=False, values=hyper_params.mnist_values)

    # Loss function
    loss_function = nn.MSELoss()
    # Optimizers
    optimizer = optim.SGD([{'params': model.features.hidden_layer.parameters()},
                              {'params': model.readout.parameters(),
                               'lr': hyper_params.regular_lr}],
                             lr=hyper_params.slow_lr)

    # Creating 'empty' arrays for future storing of accuracy metrics
    accuracy = np.zeros((hyper_params.epochs, 3))

    print("Training models...")
    for epoch in range(hyper_params.epochs):

        train(train_loader, device, model, loss_function, optimizer, values=hyper_params.mnist_values)

        accuracy[epoch][0] = epoch + 1
        accuracy[epoch][1] = check_accuracy(device, model, train_loader, hyper_params.mnist_values).cpu()
        accuracy[epoch][2] = check_accuracy(device, model, validate_loader, hyper_params.mnist_values).cpu()
        # CALCULATE THE K.A. AND RECORD IT TO A CSV (FOR SLOW MODEL)
        print("Slow: ")
        print(accuracy[epoch])

        # compute al. on both t and v.

    # Saving the entire model
    save_path = 'model_save'
    i = 0
    while os.path.exists(save_path):
        save_path = 'model_save' + str(i)

    torch.save(model.state_dict(), '../' + save_path + '.pt')
