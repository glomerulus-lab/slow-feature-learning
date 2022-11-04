import torch  # Base torch library
from torch.utils.data import DataLoader  # Minibathces
import torchvision.datasets as datasets  # MNIST dataset
import torchvision.transforms as transforms
import numpy as np
from torch.nn.functional import one_hot


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

    # Set array full of zeros.
    kernel_alignments = torch.zeros(len(loader))

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.reshape(data.shape[0], -1).to(device=device)
        targets = targets.to(device=device)

        # Forwards.
        scores = model(data)
        labels = one_hot(targets.long() % len(values)).to(torch.float32)
        loss = loss_function(scores, labels)

        # Backwards.
        optimizer_function.zero_grad()
        loss.backward()

        optimizer_function.step()
        phi = model.features(data)

        kernel_alignments[batch_idx] = kernel_calc(targets, phi)

    return torch.mean(kernel_alignments).item()
    # return mean and STD or STE of kernel alignment


def record_accuracy(device, model, train_loader, test_loader, epoch, ste, mean, values=list(range(10))):
    epoch_accuracy = np.array([[
        epoch + 1,
        check_accuracy(device, model, train_loader, values).cpu(),
        check_accuracy(device, model, test_loader, values).cpu(),
        mean,
        ste
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

    return float(num_correct / num_samples)


def classify_targets(targets, values):
    new_targets = targets.clone()

    # Changing targets to a classifiable number.
    for key, element in enumerate(values):
        new_targets[targets == element] = key
    return new_targets


# Kernel Alignment Fucntions

def kernel_calc(y, phi):

    # Output Kernel
    y = torch.t(torch.unsqueeze(y, -1)).float()
    K1 = torch.matmul(torch.t(y), y).to('cuda')
    K1c = kernel_centering(K1.float())

    # Feature Kernel
    K2 = torch.mm(phi, torch.t(phi))
    K2c = kernel_centering(K2)

    return kernel_alignment(K1c, K2c)


def frobenius_product(K1, K2):
    return torch.trace(torch.mm(K2, torch.t(K1)))


def kernel_alignment(K1, K2):
    return frobenius_product(K1, K2) / ((torch.norm(K1, p='fro') * torch.norm(K2, p='fro')))


def kernel_centering(K):
    # Lemmna 1

    m = K.size()[0]
    I = torch.eye(m)
    l = torch.ones(m, 1)

    # I - ll^T / m
    mat = I - torch.matmul(l, torch.t(l)) / m

    return torch.matmul(torch.matmul(mat, K), mat)


def ones(vector):
    for i in range(vector.size()[1]):
        if vector[0][i] == 9:
            vector[0][i] = 1
        elif vector[0][i] == 8:
            vector[0][i] = -1
    return vector
