#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as f


class Linear(nn.Module):
    """
    Model using linear layers
    """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=200):
        super(Linear, self).__init__()
        
        hidden_dim2 = int(hidden_dim / 2)
        
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim2)
        self.linear3 = torch.nn.Linear(hidden_dim2, output_dim)
    
    def forward(self, x):
        x = f.leaky_relu(self.linear1(x))
        x = f.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        
        return x