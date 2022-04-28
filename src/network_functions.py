import torch  # Base torch library
from torch.utils.data import DataLoader  # Minibathces
import torchvision.datasets as datasets  # MNIST dataset
import torchvision.transforms as transforms
import numpy as np


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def mnist_dataset(batch_size, train=True, values=list(range(10))):
    # Initializing MNIST data set.
    dataset = datasets.MNIST(root='dataset/', train=train, transform=transforms.ToTensor(), download=True)

    targets_list = dataset.targets.tolist()
    values_index = [i for i in range(len(dataset)) if targets_list[i] in values]

    # Creating a subset of ### MNIST targets.
    subset = torch.utils.data.Subset(dataset, values_index)
    loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)

    return loader


def train(loader, device, model, loss_function, optimizer_function, values=list(range(10))):
    # Training on each data point.
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.reshape(data.shape[0], -1).to(device=device)
        targets = targets.to(device=device)

        # Forwards.
        scores = model(data)
        loss = loss_function(scores, classify_targets(targets, values))

        # Backwards.
        optimizer_function.zero_grad()
        loss.backward()

        optimizer_function.step()


def record_accuracy(device, model, train_loader, test_loader, epoch, values=list(range(10))):
    epoch_accuracy = np.array([[
        epoch + 1,
        check_accuracy(device, model, train_loader, values).cpu(),
        check_accuracy(device, model, test_loader, values).cpu()
    ]])

    return epoch_accuracy


def check_accuracy(device, model, loader, values=list(range(10))):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = classify_targets(y, values).to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64images x 10,

            predictions = scores.argmax(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return 100 - 100. * num_correct / num_samples


def classify_targets(targets, values):
    new_targets = targets.clone()

    # Changing targets to a classifiable number.
    for key, element in enumerate(values):
        new_targets[targets == element] = key
    return new_targets
