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

if __name__ == '__main__':

	# Setting model to train on GPU (if available).
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	hyper_params = {
		"input_size": 784,
		"middle_width": 10,
		"classes": 2,
		"learning_rate": 0.1,
		"readout_learning_rate": 0.1,
		"batch_size": 10,
		"epochs": 4 
	}

	# Initialling model
	model = NN(middle_width=hp["middle_width"],
			   classes=hp["classes"])

	mnist_values = []

	# TODO TURN TO A FUNCTION
	train_loader = mnist_dataset(hp["batch_size", True, mnist_values])
	val_loader = mnist_dataset(hp["batch_size", False, mnist_values])

	# Loss function 
	loss_function = nn.MSELoss()

	# Optimizer
	optimizer = optim.SGD([{"params": model.features.hidden_layer.parameters()},
							{"params": model.readout.parameters(),
							"lr": hp["learning_rate"]}],
							lr= hp["readout_learning_rate"])
	
	# Training Model
	for epoch in range(hp["Epochs"]):

		printf("Epoch: {epoch}")

		# TODO CREATE CHECKPOINT TO STORE MODEL

		for batch_idx, (data, targets) in enumerate(train_loader):
			data = data.reshape(data.shape[0], -1).to(device = device)
			labels = classify(targets, values)

			# Forwards pass.
			scores = model(data)
			loss = loss_function(scores, labels)

			# Backwards pass.
			optimizer.zero_grad()
			loss.backward()

			optimizer.step()