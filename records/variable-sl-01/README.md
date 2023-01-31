# Variable Slow Learning  Rate Experiment

This experiment attempts to demonstrate the effect that varying readout learning rates have on a model's performance.
 Thirteen distinct readout rates were tested, ranging from 0.0001 to 0.1 in increments of 2.5x10^n. Additonally the
 model was trained on the MNIST dataset provided by the PyTorch library. Further explanation for the hyper-parameters
 are described below.

### Hyper-Parameters
- Hidden Layer Size: 2000
- Epochs: 3000
- Batch Size: 200
- Learning Rate = 0.1
- Readout Learning Rate = 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1

### Data
Trained on the MNIST dataset filter for digits of 0 and 1. 