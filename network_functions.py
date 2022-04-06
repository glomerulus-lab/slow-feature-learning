import torch  # Base torch library
from torch.utils.data import DataLoader  # Minibathces
import torchvision.datasets as datasets  # MNIST dataset
import torchvision.transforms as transforms
import numpy as np


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def mnist_dataset(batch_size, train=True):
    dataset = datasets.MNIST(root='dataset/', train=train, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
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

def memory():
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    f"Parameters memory: {mem_params}"
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    f"Buffer memory: {mem_bufs}"
    mem = mem_params + mem_bufs  # in byte
    f"Total memory: {mem}"