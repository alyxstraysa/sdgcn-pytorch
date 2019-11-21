#Layers File
import numpy as np
import torch
import torch.nn.functional as F

#dynamic rnn layer
def dynamic_rnn(cell, inputs, n_hidden, length, max_len, out_type='last'):
    outputs, h_n = torch.nn.RNNCell(
        
    )


def WXbA_Relu(X, A, W, b):
    """
    :param W: (600,600)
    :param X:  (?,600,targets_num)
    :param A: (targets_num,targets_num)
    :param b:
    :return:
    """
    
    X_shape_1 = X.shape[1]
    X_shape_2 = X.shape[2]
    X_shape_1_ = W.shape[1]
    X_trans = torch.transpose(X, 1, 2)
    W_X_trans = torch.einsum('ijk, kl-> ijl', X_trans, W)
    W_X_b_trans_reshape = W_X_trans.view([-1, X_shape_1_]) + b
    W_X_b_trans = W_X_b_trans_reshape.view([-1, X_shape_2, X_shape_1_]) #(?, targets_num, 600)
    W_X_b = torch.transpose(W_X_b_trans, 1, 2) #(?,600,targets_num)
    W_X_b_A_relu = F.relu(torch.matmul(W_X_b, A))

    return W_X_b_A_relu
    
