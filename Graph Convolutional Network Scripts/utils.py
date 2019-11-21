import numpy as np
import torch

def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = length.view(-1, 1).float()
    inputs = torch.sum(inputs, 1, keepdim=False) / length
    
    return inputs