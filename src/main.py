from network_functions import *
from network import *
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import pandas as pd
import sys

if __name__ == '__main__':
    # Checking & Setting Device Allocation
    device = set_device()
    print(f"Running on {device}")

    # Hyper Parameters
    hp = {
        "Input Size": 784,
        "Middle Layer Width": 2000,
        "Num Classes": 2,
        "Regular Learning Rate": 0.01,
        "Slow Learning Rate": 0.001,
        "Batch Size": 2000,
        "Epochs": 3000
    }
    print(f"Hyper Parameters: {hp}")

    # Initializing Model
    slow_model = NN(input_size=hp["Input Size"],
                    middle_width=hp["Middle Layer Width"],
                    num_classes=hp["Num Classes"]).to(device=device)

    reg_model = NN(input_size=hp["Input Size"],
                   middle_width=hp["Middle Layer Width"],
                   num_classes=hp["Num Classes"]).to(device=device)

    # Loading MNIST Dataset
    mnist_values = list(map(int, list(sys.argv[1])))
    print(f"MNIST digits {mnist_values}")
    train_loader = mnist_dataset(hp["Batch Size"], values=mnist_values)
    validate_loader = mnist_dataset(hp["Batch Size"], train=False, values=mnist_values)

    # Loss function
    loss_function = nn.MSELoss()

    # Optimizers
    sl_optimizer = optim.SGD([{'params': slow_model.features.hidden_layer.parameters()},
                              {'params': slow_model.readout.parameters(),
                               'lr': hp["Regular Learning Rate"]}],
                             lr=hp["Slow Learning Rate"])
    r_optimizer = optim.SGD(reg_model.parameters(), lr=hp["Regular Learning Rate"])

    # Creating 'empty' arrays for future storing of accuracy metrics
    slow_accuracy = np.zeros((hp["Epochs"], 4))
    regular_accuracy = np.zeros((hp["Epochs"], 4))

    print("Training models...")
    for epoch in range(hp["Epochs"]):

        # Slow Model
        sl_mean = train(train_loader, device, slow_model, loss_function, sl_optimizer, values=mnist_values)
        slow_accuracy[epoch][0] = epoch
        slow_accuracy[epoch][1] = check_accuracy(device, slow_model, train_loader, mnist_values)
        slow_accuracy[epoch][2] = check_accuracy(device, slow_model, validate_loader, mnist_values)
        slow_accuracy[epoch][3] = sl_mean
        # CALCULATE THE K.A. AND RECORD IT TO A CSV (FOR SLOW MODEL)
        print("Slow: ")
        print(slow_accuracy[epoch])

        # Regular Model
        reg_mean = train(train_loader, device, reg_model, loss_function, r_optimizer, values=mnist_values)
        regular_accuracy[epoch][0] = epoch + 1
        regular_accuracy[epoch][1] = check_accuracy(device, reg_model, train_loader, mnist_values)
        regular_accuracy[epoch][2] = check_accuracy(device, reg_model, validate_loader, mnist_values)
        regular_accuracy[epoch][3] = reg_mean


        # CALCULATE THE K.A. AND RECORD IT TO A CSV (FOR REG MODEL)
        print("Reg: ")
        print(regular_accuracy[epoch])
        print(f"-Finished epoch {epoch + 1}/{hp['Epochs']}")

        # compute al. on both t and v.

    # Accuracy csv
    complete_array = np.concatenate((slow_accuracy, regular_accuracy), axis=1)
    complete_dataframe = pd.DataFrame(complete_array).to_csv('../accuracy_metrics'+sys.argv[1]+sys.argv[2])
    print(f"-Saved accuracy metrics as 'accuracy_metrics{str(mnist_values)}'")

    # Saving the entire model
    torch.save(slow_model.state_dict(), '../slow_model.pt')
    print(f"-Saved Regular Model Parameters as 'slow_model.pt'")
    torch.save(reg_model.state_dict(), '../reg_model.pt')
    print(f"-Saved Regular Model Parameters as 'reg_model.pt'")