# Main function for MNist NN Model
#
# Trains on specified MNIST values, and can train for both a split lr model and regular.
#
# By Cameron G. Kaminski

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import sys
import json 

def store_data(dict, mnist_digits, lr, slr):
  digits = ""
  for v in mnist_digits:
    digits = digits + str(v)
  filename = digits
  with open(filename + ".json", "w") as write_file:
    json.dump(dict, write_file)

  

if __name__ == '__main__':

  # Hyper parameters
  print(sys.argv[0])
  #mnist_values = list(map(int, map(float, list(sys.argv[0]))))
  mnist_values = [1,2]
  middle_width = int(sys.argv[1])
  epochs = int(sys.argv[2])
  batch_size = int(sys.argv[3])
  lr = sys.argv[4]
  if sys.argv[5] is None: 
    slr = lr
  else:
     slr = sys.argv[5]
  
  # Setting the device to use cuda if avaiable. 
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # initializing the model
  model = resources.NN(20, 10, values = [2, 7]).to(device)
  # initializing the dataframe
  training = resources.mnist_dataset(5, values = [2, 7])
  val = resources.mnist_dataset(5, train=False, values= [2, 7])
  # optimizer
  optimizer = optim.SGD([{'params': model.features.hidden_layer.parameters()},
                            {'params': model.readout.parameters(),
                             'lr': 0.1}],
                           lr=0.01)
  # loss 
  loss = nn.MSELoss()

  # Training the model
  cka, train_accuracy, val_accuracy = model.trains(training, val, loss, optimizer)
  
  data = {
    "Centered Kernel Alignment": cka.tolist(),
    "Training Accuracy": train_accuracy.tolist(),
    "Validation Accuracy": val_accuracy.tolist()
  }

  store_data(data, mnist_values, lr, slr)