import torch  # Base torch library
from torch.utils.data import DataLoader  # Minibathces
import torchvision.datasets as datasets  # MNIST dataset
import torchvision.transforms as transforms
import numpy as np

def mnist_dataset(batch_size, train=True, values=list(range(10))):
    # Initializing MNIST data set.
    dataset = datasets.MNIST(root='dataset/', train=train, transform=transforms.ToTensor(), download=True)

    targets_list = dataset.targets.tolist()
    values_index = [i for i in range(len(dataset)) if targets_list[i] in values]

    # Creating a subset of ### MNIST targets.
    subset = torch.utils.data.Subset(dataset, values_index)
    loader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)

    return loader

def classify_targets(targets, values):
    new_targets = targets.clone()

    # Changing targets to a classifiable number.
    for key, element in enumerate(values):
        new_targets[targets == element] = key
    return new_targets

def check_accuracy(model, loader, mnist_values, device):
    correct = 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = classify_targets(y, mnist_values).to(device=device)
            x = x.reshape(x.shape[0], -1)

            # 64images x 10
            scores = model(x)

            predictions = scores.argmax(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)
    return correct / samples
