import torch  # Base torch library
from torch.utils.data import DataLoader  # Minibathces
import torchvision.datasets as datasets  # MNIST dataset
import torchvision.transforms as transforms
import numpy as np


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def mnist_dataset(batch_size, train=True, values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    dataset = datasets.MNIST(root='dataset/', train=train, transform=transforms.ToTensor(), download=True)
    list = []
    targets_list = dataset.targets.tolist()
    for i in range(len(dataset)):
        if targets_list[i] in values:
            list.append(i)
    dataset1 = torch.utils.data.Subset(dataset, list)
    loader = DataLoader(dataset=dataset1, batch_size=batch_size, shuffle=True)
    return loader


def train(loader, device, model, loss_function, optimizer_function):
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.reshape(data.shape[0], -1).to(device=device)
        targets = targets.to(device=device)

        # Forwards.
        scores = model(data)
        loss = loss_function(scores, targets)

        # Backwards.
        optimizer_function.zero_grad()
        loss.backward()

        optimizer_function.step()


def record_accuracy(device, model, train_loader, test_loader, epoch):
    epoch_accuracy = np.array([[
        epoch+1,
        check_accuracy(device, model, train_loader),
        check_accuracy(device, model, test_loader)
    ]])

    return epoch_accuracy


def check_accuracy(device, model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64images x 10,

            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return 100 - float(num_correct) / float(num_samples) * 100
