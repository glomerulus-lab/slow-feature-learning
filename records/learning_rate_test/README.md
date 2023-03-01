# 11152022 Learning Rate Testing 

MNIST data model was run on 4 different digits (1 0, 2 7, 4 6, 8 9). Additionally each digit combination was run for four seperate learning rates. 

*note that in each model the learning rate for the output was set to a size < the learning rate of the rest of the model.*

The set learning rates are as listed:

- lr = 0.1 and slwlr = 0.01 *saved as 'rXX.json'*

- lr = 0.01 and slwlr = 0.001 *saved as 'rXX(1).json'*

- lr = 0.9 and slwlr = 0.09 *saved as 'rXX(1)(1).json'*

- lr = 0.2 and slwlr = 0.1 *saved as 'rXX(1)(1)(1).json'*

The primary purpose of this running was to compare the learning rates for the different mnist digits to estimate the best learning rate for the model.

