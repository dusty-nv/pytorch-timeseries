#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    GRU RNN model
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=32, num_layers=2):
        super(GRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        
        # UserWarning: RNN module weights are not part of single contiguous chunk of memory
        #self.gru.flatten_parameters()   
        
        x, (hn) = self.gru(x, (h0.detach()))
        x = self.fc1(x[:, -1, :]) 
        
        return x