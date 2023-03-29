# 
# LOAD PYTORCH MODEL 
# BY : CAMERON KAMINSKI
# 

import torch
import numpy as np
import resources as rs

def load_model(lr : float, slr : float, digits : (int, int), epoch : int):
    """
    Takes model specifications (i.e. learning rates and digit pair), then loads
    the pytorch model for the specified epoch.

    :param lr: learning rate
    :param slr: slow learning rate
    :param digits: digits used on model
    :param epoch: epoch that model was saved on
    :return: PyTorch model for the specifed model
    :rtype: torch.nn
    """

    PATH = "/model-saves" + 
        str(lr) + '_' + str(slr) + '_' + str(digits(0)) + str(digits(1)) +
        "{:04d}".format(epoch)

    model_state = torch.load(PATH)
    model = rs.NN(input_size=784, middle_width=2048, num_classes=2)

    return model
    

