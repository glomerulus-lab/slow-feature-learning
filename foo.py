import torch
ca

dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
print(dataset)
loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
print(loader)
