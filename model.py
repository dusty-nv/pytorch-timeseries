#!/usr/bin/env python3
# coding: utf-8

import os
import math
import torch

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from dataset import Dataset, DataSubset, DataLoader
from models import create_model, model_list

torch.manual_seed(1)  # reproducibility


class Model:
    """
    Time-series forecasting/classification model.
    """
    def __init__(self, model, input_dim=-1, output_dim=-1, **kwargs):
        """
        Either create a new model or load a previously-trained model from a checkpoint.
        
        Parameters:
          model -- String indicating the model architecture name (i.e. 'gru', 'cnn_gru', 'linear')
                   or a .pth checkpoint to load, from which the other arguments will be loaded too.
          input_dim -- The number of input features to the model. This is the last dimension of the input tensor.
                       Required for new models that aren't being loaded from checkpoint.
          output_dim -- The number of outputs from the model.
                        Required for new models that aren't being loaded from checkpoint.
          kwargs -- Hyperparameters for new models that are passed to the model constructor
        """
        ext = os.path.splitext(model)[1]

        if len(ext) == 0:
            kwargs['input_dim'] = input_dim
            kwargs['output_dim'] = output_dim
            
            self.model = create_model(model, **kwargs).cuda()
        else:
            raise NotImplementedError('loading from checkpoint not yet implemented')
            
    def train(self, dataset, epochs=1000, batch_size=-1, learning_rate=0.05, scheduler='StepLR_250'):
        """
        Train the model on the provided dataset.
        
        Parameters:
            dataset (dataset.Dataset) -- the loaded dataset 
            epochs (int) -- the number of epochs to train for
            batch_size (int) -- the batch size to use (default is entire dataset)
            learning_rate (float) -- the initial learning rate
            scheduler -- the learning rate scheduler to use (StepLR or ReduceLROnPlateau). These strings can have a suffix of the
                         form '_N', where N is the step epochs parameter used by the scheduler.  For example, 'StepLR_30' will reduce
                         the learning rate every 30 epochs.  'ReduceLROnPlateau_10' will reduce when there is a 10-epoch plateau.
        
        TODO add plateau parameter, which makes epochs argument mean the number of epochs to plateu before stopping
        """
        dataloaders = {}
        
        for subset in dataset.subsets:
            dataloaders[subset] = DataLoader(dataset.subsets[subset], batch_size=batch_size)
     
        if 'train' not in dataloaders:
            raise ValueError("dataset requires 'train' subset for training")
             
        if 'val' not in dataloaders:
            raise ValueError("dataset requires 'val' subset for training")
                
        # create optimizer, scheduler, and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)   
        scheduler = self._create_scheduler(scheduler, optimizer)
        criterion = self._create_loss(dataset.classification)

        # training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for i, (input, target) in enumerate(dataloaders['train']):
                output = self.model(input)
                loss = criterion(output, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
     
                train_loss += loss.item() * dataloaders['train'].last_batch_weight

            # eval model
            val_loss = self.eval(dataloaders['val'], criterion)
            print(f"Epoch {epoch:03d}  LR={scheduler._last_lr[0]:.2g}  train_loss={train_loss:.8f}  val_loss={val_loss:.8f}")
            
            # update learning rate
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=train_loss)
            else:
                scheduler.step()

        return train_loss, val_loss
    
    def eval(self, data, criterion=None, return_outputs=False):
        """
        Eval model on dataset, return the loss
        """
        if isinstance(data, DataLoader):
            dataloader = data
        elif isinstance(data, DataSubset):
            dataloader = DataLoader(data)
        elif isinstance(data, Dataset):
            dataloader = Dataset.subsets.get('val', Dataset.subsets.get('test', Dataset.subsets.get('train')))
            
        if not criterion:
            criterion = self._create_loss(dataloader.dataset.parent.classification)
            
        self.model.eval()
        
        loss = 0.0
        outputs = []
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataloader):
                output = self.model(input)
                loss += criterion(output, target).item() * dataloader.last_batch_weight
                
                if return_outputs:
                    outputs.append(output)
                    
        if return_outputs:
            return loss, torch.cat(outputs)
        else:
            return loss
            
    @staticmethod
    def _create_loss(classification):
        if classification:
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()
            
        criterion.cuda()
        return criterion
        
    @staticmethod
    def _create_scheduler(scheduler, optimizer):
        """
        Create a scheduler from a param string like 'StepLR_30'
        """
        if scheduler.startswith('StepLR'):
            return StepLR(optimizer, step_size=Model._parse_param(scheduler, default=250))
        elif scheduler.startswith('ReduceLROnPlateau'):
            return ReduceLROnPlateau(optimizer, patience=Model._parse_param(scheduler, default=10))
        else:
            raise ValueError(f"invalid scheduler '{scheduler}'") 
        
    @staticmethod
    def _parse_param(str, default):
        """
        Parse a parameter in a string of the form 'text_value'
        """
        idx = str.find('_')
        
        if idx < 0 or idx == (len(str) - 1):
            return default

        return int(str[idx+1:])
        
    @staticmethod
    def available_models():
        """
        Returns the list of model architectures
        """
        return model_list
        