#
#
#
#
#


from network_functions import *
import os
from argparser import parseHyperparams
from network import *
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import datetime
import time

if __name__ == '__main__':
    start_time = time.time()
    # Checking & Setting Device Allocation
    device = set_device()

    # Parsing Hyperparameters from the system arguments.
    hyper_params = parseHyperparams()

    # Creating the model save directory
    print(os.getcwd())
    dir_path = str("model-saves/" + str(hyper_params.regular_lr) + '_'
               + str(hyper_params.slow_lr) + '_' + str(hyper_params.mnist_values[0])
               + str(hyper_params.mnist_values[1]))
    j = 0
    while os.path.exists(dir_path):
        dir_path += str(j)
    os.mkdir(dir_path)

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

    # Training the model over each epoch
    for epoch in range(hyper_params.epochs):

        # Training the model over each batch
        train(train_loader, device, model, loss_function, optimizer,
              values=hyper_params.mnist_values)

        # Saving the entire model
        torch.save(model.state_dict(), dir_path + "/model" + '{:04d}'.format(epoch) + ".pt")

    end_time = time.time()

    # Creating a .txt file to store the hyperparameters.
    with open(dir_path + '/hyperparams.txt', 'w') as f:
        f.write("Hyperparameters: " +
                str({hyper_param: getattr(hyper_params, hyper_param) for hyper_param in vars(hyper_params)}))
        f.write("\nDATE TIME : " + str(datetime.datetime.now()))
        f.write("\nRUNTIME: " + str(end_time - start_time))