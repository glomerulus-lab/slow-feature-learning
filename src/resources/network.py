# Network Class
#
# Contains the NN class for the slow-feature-learning model.
#
# By Cameron Kaminski

import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from torch.nn.functional import one_hot

"""
Nueral Network Class

@param input_size: # of features given from data (in case of MNIST its 28x28=784).
@param middle_width: # of hidden layer nodes in NN.
@param num_classes: # of unique classes for NN to make predictions on.
"""
class NN(nn.Module):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, middle_width, epochs, values):

        super(NN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('hidden_layer', nn.Linear(784, middle_width)),
            ('hidden_activation', nn.ReLU()),
        ]))
        self.readout = nn.Linear(middle_width, len(values))

        # Hyper Parameters
        self.epochs = epochs
        self.values = values

    # Forward pass.
    def forward(self, x):
        return self.readout(self.features(x))

    def train_one_epoch(self, loader, loss_function, optimizer, record=True):

        # Array of centered kernal analysis. 
        cka = torch.zeros(len(loader))

        for i, (data, targets) in enumerate(loader):
            data = data.reshape(data.shape[0], -1).to(device=self.device)
            targets = targets.to(torch.float32).to(device=self.device)

            # Forwards pass.
            scores = self(data)

            labels = one_hot(targets.long() % len(self.values)).to(torch.float32)
            output = loss_function(scores, labels)

            # Backwards Pass.
            optimizer.zero_grad()
            output.backward()

            # Step. 
            optimizer.step()
            
            # Recording the C.K.A. for the batch index.
            if record:
                cka[i] = kernel_calc(self.device, targets, self.features(data).to(device=self.device))
        
        # Returning the C.K.A. if the option to record was chosen.
        if record:
            return cka


    def trains(self, training, val, loss_function, optimizer, recordcka=True):
        # Array full of the mean C.K.A. across the the model.
        mcka = torch.zeros(self.epochs).to(device=self.device)
        train_accuracy = torch.zeros(self.epochs)
        val_accuracy = torch.zeros(self.epochs)
        for epoch in range(self.epochs):
            # When recordingm run the training & return the C.K.A. 
            if recordcka:
                mcka[epoch] = torch.mean(self.train_one_epoch(training, loss_function, optimizer)).item()
                train_accuracy[epoch] = self.check_accuracy(training)
                val_accuracy[epoch] = self.check_accuracy(val)

                print(f"Epoch {str(epoch)} | {round(mcka[epoch].item(), 5)} | {round(train_accuracy[epoch].item(), 5)} | {round(val_accuracy[epoch].item(), 5)}")
            # No record case.
            else: train_one_epoch(training, loss_function, optimizer, record=False)
        
        if recordcka:
            return mcka, train_accuracy, val_accuracy
        else: 
            return train_accuracy, val_accuracy

    def classify_targets(self, targets):
        new_targets = targets.clone()
    
        # Changing targets to a classifiable number.
        for key, element in enumerate(self.values):
            new_targets[targets == element] = key
        return new_targets
    
 
    def check_accuracy(self, loader):
        correct = 0 
        samples = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = self.classify_targets(y).to(device=self.device)
                x = x.reshape(x.shape[0], -1)
                
                # 64images x 10
                scores = self(x)

                predictions = scores.argmax(1)
                correct += (predictions == y).sum()
                samples += predictions.size(0)
        return correct / samples

def kernel_calc(device, y, phi):

    # Output Kernel
    y = torch.t(torch.unsqueeze(y, -1)).to(device=device)
    K1 = torch.matmul(torch.t(y), y).to(device=device)
    K1c = kernel_centering(device, K1.float()).to(device=device)

    # Feature Kernel
    K2 = torch.mm(phi, torch.t(phi)).to(device=device)
    K2c = kernel_centering(device, K2).to(device=device)

    return kernel_alignment(K1c, K2c).to(device=device)


def frobenius_product(K1, K2):
    return torch.trace(torch.mm(K2, torch.t(K1)))


def kernel_alignment(K1, K2):
    return frobenius_product(K1, K2) / ((torch.norm(K1, p='fro') * torch.norm(K2, p='fro')))


def kernel_centering(device, K):
    # Lemmna 1

    m = K.size()[0]
    I = torch.eye(m).to(device=device)
    l = torch.ones(m, 1).to(device=device)

    # I - ll^T / m
    mat = I - torch.matmul(l, torch.t(l)).to(device=device)/ m

    return torch.matmul(torch.matmul(mat, K), mat)
