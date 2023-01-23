# Differing Learing Rates Experiment

Hyper Parameters:
	- MNIST-DIGITS (0, 1)
	- MIDDLE-LAYER-WIDTH 2000
	- EPOCHS 3000
	- BATCH-SIZE 200
	- LEARNING-RATE 0.1
	- SLOW-LEARNING-RATE [0.1, 0.075, 0.05, 0.025, 0.01, ..., 0.0001]

Exerpiement Description:
	
Trained models on the same input learning rate (and other hyperparameters) but changed the output learning rate by increments of 2.5*10^{-x}.

