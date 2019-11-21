#GCN BERT (PyTorch)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import reduce_mean_with_len
from layers import WXbA_Relu

#shape of train = 3608 rows, len of 79 in each row (that's the max length of the sentence in train)
#shape of train_target = (3608, 23, 768) where 3608 is number of sentences, 23 is the max length of target, 768
#is the word embedding dimension of BERT

class GCN_Bert(nn.Module):
    def __init__(self, sequence_length, target_sequence_length,targets_num_max, num_classes, word_embedding_dim, l2_reg_lambda=0.0,
                 num_hidden=100):
        super(GCN_Bert, self).__init__()

        #embedding layer
        self.embedded_sen = F.dropout(self.input_x, p = self.dropout_keep_prob)

        self.embedded_target = F.dropout(self.input_target, p =self.dropout_keep_prob)

        self.embedded_targets_all = list(range(targets_num_max))
        for i in range(targets_num_max):
            self.embedded_target_i = F.dropout(self.input_targets_all[:, i, :, :], p = self.dropout_keep_prob)
            self.embedded_targets_all[i] = self.embedded_target_i #13 * (?, 21, 300) 

        ###LSTM layer###
        #Bidirectional LSTM for context

        num_hidden = 300

        self.LSTM_hidden_sen = nn.LSTM(self.embedded_sen, num_hidden, )
        pool_sen = reduce_mean_with_len(self.LSTM_Hiddens_sen, self.sen_len)

        #Bidirectional LSTM for targets
        self.LSTM_targets_all = list(range(targets_num_max))
        poor_targets_all = list(range(targets_num_max))


        ###Attention Layer###

        ###GCN Layer###

        #GCN Layer 1
        GCN1_cross = WXbA_Relu(self.targets_concat,self.relate_cross,W_cross,b_cross)
        GCN1_self = WXbA_Relu(self.targets_concat,self.relate_self,W_self,b_self)
        GCN1_out = GCN1_cross + GCN1_self     #(?,600,13)

        #GCN Layer 2
        GCN2_cross = WXbA_Relu(GCN1_out, self.relate_cross, W_cross, b_cross)
        GCN2_self = WXbA_Relu(GCN1_out,self.relate_self,W_self,b_self)
        GCN2_out = GCN2_cross + GCN2_self        #(?,600,13)

        target_which = torch.unsqueeze(self.target_which, 1) # (?,1,13)
        self.GCN2_out = torch.mul(GCN2_out, target_which) #(?,600,13)*(?,1,13) = (?,600,13)
        self.targets_representation = torch.sum(self.GCN2_out, 2)  # (?,600)

    def forward(self, inputs):
        pass


gcnbert = GCN_Bert().to_device('Cuda')

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gcnbert.params(), lr = learning_rate)

#run the forward pass
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = gcnbert(inputs)
        loss = criterion(outputs, labels)

        #Print the accuracy
        correct_pred = torch.equal(outputs, labels)
        accuracy = sum(correct_pred) / len(correct_pred)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Fully connected neural network with one hidden layer example
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out