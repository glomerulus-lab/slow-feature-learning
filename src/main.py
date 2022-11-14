# Main function for MNist NN Model
#
# Trains on specified MNIST values, and can train for both a split lr model and regular.
#
# By Cameron G. Kaminski

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import resources
import sys
import json 

def store_data(model, dict, mnist_digits, lr, slr):
  digits = ""
  for v in mnist_digits:
    digits = digits + str(v)
  if lr != slr:
    filename = "s" + digits
  else:
    filename = 'r' + digits
  with open("records/" + filename + ".json", "w") as write_file:
    json.dump(dict, write_file)
  torch.save(model.state_dict(), "models/" + filename)

  

if __name__ == '__main__':

  # Hyper parameters
  print(sys.argv[1])
  mnist_values = list(map(int, list(sys.argv[1])))
  middle_width = int(sys.argv[2])
  epochs = int(sys.argv[3])
  batch_size = int(sys.argv[4])
  lr = float(sys.argv[5])
  if sys.argv[6] is None: 
    slr = lr
  else:
     slr = float(sys.argv[6])
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # initializing the model
  model = resources.NN(middle_width, epochs, mnist_values).to(device=device)
  # initializing the dataframe
  training = resources.mnist_dataset(batch_size, values = mnist_values)
  val = resources.mnist_dataset(batch_size, train=False, values = mnist_values)
  # optimizer
  optimizer = optim.SGD([{'params': model.features.hidden_layer.parameters()},
                            {'params': model.readout.parameters(),
                             'lr': lr}],
                           lr=slr)
  # loss 
  loss = nn.MSELoss()

  # Training the model
  cka, loss_values, train_accuracy, val_accuracy = model.trains(training, val, loss, optimizer)
  
  data = {
    "Centered Kernel Alignment": cka.tolist(),
    "Loss": loss_values.tolist(),
    "Training Accuracy": train_accuracy.tolist(),
    "Validation Accuracy": val_accuracy.tolist()
  }

  store_data(model, data, sys.argv[1], lr, slr)
