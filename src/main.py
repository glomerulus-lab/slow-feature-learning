"""
Training Script For Slow Feature Learning Project
Author: Cameron Kaminski
"""
from resources import *
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from torch.nn.functional import one_hot

if __name__ == '__main__':

    # Checking & Setting Device Allocation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper params..
    hp = read_hyperparams('hyper-parameters/hyperparams.txt')

    # Initializing models:
    model = NN(hp['input'], hp['ml_width']).to(device)

    # Loading MNist dataset
    mnist_values = [0, 1]
    train = mnist_dataset(hp['batch_size'], values=mnist_values)
    val = mnist_dataset(hp['batch_size'], values=mnist_values)

    # Loss function
    loss = nn.MSELoss()

    # Optimizers
    optimizer = optim.SGD([{'params': model.features.hidden_layer.parameters()},
                           {'params': model.readout.parameters(),
                            'lr': hp['learning_rate']}],
                          lr=hp['slow_learning_rate'])

    loss_function = nn.MSELoss()

    # Training the Model.
    print(f"Training the model on {device} for {mnist_values} with HP: /n"
          f"learning rate = {hp['learning_rate']}\n"
          f"slow learning rate = {hp['slow_learning_rate']}\n"
          f"batch size = {hp['batch_size']}\n"
          f"epochs = {hp['epochs']}\n")

    for epoch in range(hp['epochs']):

        for batch_idx, (data, targets) in enumerate(train):
            data = data.reshape(data.shape[0], -1).to(device=device)
            targets = targets.to(device=device)
            classified_targets = classify_targets(targets, mnist_values)


            # Forwards
            scores = model(data)

            labels = one_hot(classified_targets.long(), len(mnist_values)).to(torch.float32)
            loss = loss_function(scores, labels)

            # Backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model, './models/model' + str(epoch) + '.pth')
        print(f"Epoch: {epoch} | Accuracy: {check_accuracy(model, val, mnist_values, device)}")


    print("Training completed.")