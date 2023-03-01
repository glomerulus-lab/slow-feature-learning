from network_functions import *
import os
import time
from argparser import parseHyperparams
from network import *
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import pandas as pd

if __name__ == '__main__':
    # Checking & Setting Device Allocation
    device = set_device()

    # Parsing Hyperparameters from the system arguments.
    hyper_params = parseHyperparams()

    # Initializing Model
    model = NN(input_size=hyper_params.input_size,
               middle_width=hyper_params.middle_width,
               num_classes=hyper_params.num_classes).to(device=device)

    # Loading MNIST Dataset
    print(f"MNIST digits {hyper_params.mnist_values}")
    train_loader = mnist_dataset(hyper_params.batch_size,
                   values=hyper_params.mnist_values)
    validate_loader = mnist_dataset(hyper_params.batch_size, train=False,
                      values=hyper_params.mnist_values)

    # Loss function
    loss_function = nn.MSELoss()
    # Optimizers
    optimizer = optim.SGD([{
                'params': model.features.hidden_layer.parameters()},
                {'params': model.readout.parameters(),
                 'lr': hyper_params.regular_lr}],
                lr=hyper_params.slow_lr)

    for epoch in range(hyper_params.epochs):

        start_time = time.time()

        train(train_loader, device, model, loss_function, optimizer,
              values=hyper_params.mnist_values)

        end_time = time.time()

    print(f"Epoch : {epoch} || Time : {end_time - start_time}")

    # Saving the entire model
    save_path = 'model_save'
    i = 0
    while os.path.exists(save_path):
        save_path = 'model_save' + str(i)

    torch.save(model.state_dict(), '../' + save_path + '.pt')
