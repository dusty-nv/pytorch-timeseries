#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Dataset():
    """
    Tabular CSV dataset for timeseries forecasting or classifiction.
    
    The dataset is split into train/val/test subsets, which
    are stored in a dict at Dataset.subsets, where the keys
    are the name of the subset - i.e. Dataset.subsets['train']
    
    Subsets are also available at Dataset.train, Dataset.test, ect.
    And they can be indexed via dataset['train'], dataset['test'], ect.
    """
    TrainValSplit = dict(train=0.8, val=0.2)
    TrainValTestSplit = dict(train=0.7, val=0.15, test=0.15)    
    DefaultSplit = TrainValSplit
    
    TrainOnly = dict(train=1.0)
    ValOnly = dict(val=1.0)
    TestOnly = dict(test=1.0)
    
    def __init__(self, data, inputs, outputs,
                 input_scaler='standard', output_scaler='standard',
                 splits=DefaultSplit, classification=False, 
                 history=0, horizon=1, data_limit=None):
        """
        Either load a dataset from CSV, or use an existing Pandas DataFrame.
        
        Parameters:
          data (string/DataFrame) -- path to a CSV file or Pandas DataFrame
          inputs (list) -- column name/indices to use as inputs. can be a list or a single value.
                           this can also be a comma-separated string of column names or indices
          outputs (list) -- column names/indices to use as outputs. can be a list or a single value.
                            this can also be a comma-separated string of column names or indices
          input_scaler (string) -- preprocessing scaler to use for inputs ('none', 'minmax', 'standard')
          output_scaler (string) -- preprocessing scaler to use for outputs ('none', 'minmax', 'standard')
          splits (dict) -- dictionary of splits, the default is {'train' : 0.8, 'val' : 0.15}
                           see Dataset.TrainValSplit, Dataset.TrainValTestSplit, ect for pre-defined splits.
          classification (bool) -- if True, this is a classification dataset
          history (int) -- past history of inputs to use (in timesteps)
                           if history > 0, the input data will be sequenced as (N, history, num_inputs)
                           typically used for RNN/LSTM.  Otherwise, the input data will be (N, num_inputs)
          horizon (int) -- the number of timesteps into the future to forecast
                           this will shift the output data by -horizon timesteps
                           if the source data is already shifted, set this to 0
          data_limit (int) -- the number of rows of data to read (by default, all rows)
        """
        if isinstance(data, str):
            print(f"loading {data}")
            self.df = pd.read_csv(data, nrows=data_limit, parse_dates=[0])
        else:
            self.df = data

        self.subsets = {}
        self.num_classes = 0
        self.output_classes = []
        self.classification = classification
         
        self.input_scaler = self.create_scaler(input_scaler)
        self.output_scaler = self.create_scaler(output_scaler)
        
        self.df.dropna(inplace=True)
        self.select(inputs, outputs, splits, history, horizon)
        self.print_info()
        
    @property
    def train(self):
        return self.subsets['train']
    
    @property
    def val(self):
        return self.subsets['val']
     
    @property
    def test(self):
        return self.subsets['test']
       
    def __getitem__(self, subset): 
        return self.subsets[subset]
            
    def select(self, inputs=None, outputs=None, splits=None, history=None, horizon=None):
        """
        Select or change dataset parameters, such as the set of inputs and/or outputs, 
        the history length, horizon length, and/or dataset splits.
        """
        if horizon is not None:
            self.horizon = horizon
            if outputs is None:  # make sure the outputs get shifted
                outputs = self.outputs
        
        # select input features
        if inputs is not None:
            self.inputs = self.validate_columns(inputs)
            self.num_inputs = len(self.inputs)
            
            if self.num_inputs == 0:
                raise ValueError('must specify at least one valid input column name or index')
                
        # select output features
        if outputs is not None:
            self.outputs_src = self.validate_columns(outputs)
            self.num_outputs = len(self.outputs_src)
            
            if self.num_outputs == 0:
                raise ValueError('must specify at least one valid output column name or index')
                
            # if this is a classification dataset, get the number of classes
            if self.classification:
                if self.num_outputs > 0:
                    raise ValueError(f"classification datasets should only have one output - this dataset has multiple outputs {self.outputs_src}")
                self.output_classes = self.df[self.outputs_src].unique().values.tolist()
                self.num_classes = len(self.output_classes)
                
            # shift the outputs and rename output columns
            if self.horizon > 0:
                self.outputs = [output + f'{self.horizon:+d}' for output in self.outputs_src]
                self.df[self.outputs] = self.df[self.outputs_src].shift(-self.horizon)
                self.df.dropna(inplace=True)
            else:
                self.outputs = self.outputs_src
                
        # split into train/val/test
        self.split(splits, history)
            
    def split(self, splits=DefaultSplit, history=None):
        """
        Split into train/val/test subsets (by time)
        This is automatically called when the dataset it created,
        but the split allocation can be changed later.
        
        If the history parameter is specified, it will reset the 
        history length that the dataset was initially created with.
        """
        if splits is not None:
            self.splits = splits
            
        if history is not None:
            self.history = history # gets used in DataSubset.preprocess()

        # split into the specified subsets
        last_split = 0.0
        
        for type in splits:
            if type not in self.subsets:
                self.subsets[type] = DataSubset(self, type)
                
            num_samples = int(len(self.df) * splits[type])
            start_index = int(len(self.df) * last_split)
            last_split += splits[type]
            
            self.subsets[type].df = self.df[start_index : start_index + num_samples]

        # fit input/output scalers, based on the training set value distributions
        if self.input_scaler:
            self.input_scaler.fit(self.subsets['train'].df[self.inputs].values)

        if self.output_scaler:
            if self.num_classes > 0:
                print(f"warning -- classification dataset using output scaler") 
            self.output_scaler.fit(self.subsets['train'].df[self.outputs].values)

        # pre-process the data
        for subset in self.subsets:
            self.subsets[subset].preprocess()
            
    @staticmethod
    def create_scaler(type):
        if type is None:
            return None
            
        if type == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        elif type == 'standard':
            return StandardScaler()
        else:
            return None 

    @staticmethod
    def validate_columns(columns):
        """
        Verify that a list of column names / indices are valid,
        and convert indices to names for readability.
        """
        if isinstance(columns, str):
            columns = columns.split(',')
        elif isinstance(columns, int):
            columns = [columns]
            
        for idx, column in enumerate(columns):
            try:
                column_idx = int(column)
                columns[idx] = self.df.columns[column_idx]
            except ValueError:
                pass
                
        return columns

    def print_info(self):
        print('')
        print('*************************************************')
        print('** DATASET INFO')
        print('*************************************************')
        
        for subset in self.subsets:
            print('{:<13s} {:d} samples'.format('{:s}:'.format(subset), len(self.subsets[subset])))

        print('inputs:      ', len(self.inputs), self.inputs)
        print('outputs:     ', len(self.outputs), self.outputs)
        print('columns:     ', self.df.columns.values.tolist())

        if self.num_classes > 0:
            print('classes:     ', self.output_classes)
            print('class distribution:')
            for cls in self.output_classes:
                print('  [{:s}] - {:d} samples'.format(str(cls), int((self.df[self.outputs[0]] == cls).sum())))
        
        print('')
        

class DataSubset(torch.utils.data.Dataset):
    """
    Represents a train/val/test subset of the parent dataset.
    """
    def __init__(self, parent, type):
        self.df = None
        self.type = type
        self.parent = parent

    def preprocess(self):
        # convert to numpy arrays
        self.inputs = self.df[self.parent.inputs].values.astype(np.float32)
        self.targets = self.df[self.parent.outputs].values.astype(np.float32)
        
        # rescale/normalize the data
        if self.parent.input_scaler is not None:
            self.inputs = self.parent.input_scaler.transform(self.inputs)
        
        if self.parent.output_scaler is not None:
            self.targets = self.parent.output_scaler.transform(self.targets)
          
        # if single-feature outputs, flatten into 1D arrays 
        if self.parent.num_outputs == 1:
            self.targets = self.targets.ravel()

        # generate sequences
        if self.parent.history > 0:
            self.generate_sequences(self.parent.history)
            
        # create pytorch tensors
        self.inputs = torch.from_numpy(self.inputs).type(torch.FloatTensor).cuda() #.unsqueeze(1)  # seq_len=1

        if self.parent.num_classes > 0: # if this is a classification dataset, integer output data is expected
             self.targets = torch.from_numpy(self.targets).type(torch.LongTensor).squeeze().cuda()
        else:
             self.targets = torch.from_numpy(self.targets).type(torch.FloatTensor).unsqueeze(-1).cuda()
             
    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx): 
        return self.inputs[idx], self.targets[idx]

    def class_counts(self, feature=None):
        """
        Return the number of instances of each class type for the particular feature column.
        This assumes that the provided feature name does in fact have class-based labels.
        If feature is None, it will be assumed to be the first output feature of the dataset.
        """
        if feature is None:
            feature = self.parent.outputs[0]
            
        class_groups = self.df.groupby(feature)
        class_counts = []
        
        for class_label, class_df in class_groups:
            class_counts.append(len(class_df))
            #print('{:s} {:s} class {:s} - {:d} samples'.format(self.type, feature, str(class_label), len(class_df)))
        
        return class_counts
        
    def class_weights(self, feature=None):
        """
        Returns the class weights for the particular feature column, where the weight of class N is:
            max(class_counts) / class_counts[N]
        This assumes that the provided feature name does in fact have class-based labels.
        If feature is None, it will be assumed to be the first output feature of the dataset.
        """
        if feature is None:
            feature = self.parent.outputs[0]
            
        class_counts = self.class_counts(feature)
        class_weights = []
        max_count = max(class_counts)
        
        for idx, count in enumerate(class_counts):
            class_weight = max_count / count
            class_weights.append(class_weight)
            #print('{:s} {:s} class {:d} - weight {:f}'.format(self.type, feature, idx, class_weight))
            
        return class_weights
        
    def generate_sequences(self, sequence_length):
        # the last element of the sequence is the output target
        # so really the input sequences end up being 1 less
        #sequence_length += 1
        def sequence(data):
            if sequence_length == 1:
                return np.expand_dims(data,1)
                
            seq = []
            for index in range(len(data) - sequence_length): 
                seq.append(data[index : index + sequence_length]) 
                
            return np.array(seq)

        self.inputs = sequence(self.inputs)
        self.targets = sequence(self.targets)[:,-1]


class DataLoader():
    """
    Custom DataLoader for accessing DataSubset batches.
    This is used because our timeseries batch sizes are typically very high,
    and it takes PyTorch a long time to collate the single-element tensors.
    So instead, we simply return slices that are already ready to go.
    """
    def __init__(self, dataset, batch_size=-1):
        """
        dataset should be an instance of DataSubset (train/val/test)
        """
        self.dataset = dataset
        self.batch_size = batch_size

        if self.batch_size <= 0:
            self.batch_size = len(self.dataset)
            
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx): 
        if idx >= len(self):
            raise IndexError('index out-of-range')

        start_idx = idx * self.batch_size
        batch_len = min(self.batch_size, len(self.dataset) - start_idx)
        end_idx = start_idx + batch_len
        self.last_batch_weight = batch_len / len(self.dataset)
        
        return self.dataset.inputs[start_idx:end_idx], self.dataset.targets[start_idx:end_idx]

   
if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='', type=str, required=True, help='path to CSV file')
    parser.add_argument('--inputs', default='', type=str, required=True, help='name of input columns (comma-separated)')
    parser.add_argument('--outputs', default='', type=str, required=True, help='name of output columns (comma-separated)')
    parser.add_argument('--history', default=4, type=int, help='sequence history (in timesteps)')
    parser.add_argument('--horizon', default=1, type=int, help='forecasting horizon (in timesteps)')
    parser.add_argument('--input-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
    parser.add_argument('--output-scaler', default='standard', choices=['none', 'minmax', 'standard'], help='dataset preprocessing scaler to use')
    parser.add_argument('--classification', action='store_true', help='set for classification datasets')

    args = parser.parse_args()
    print(args)
    
    dataset = Dataset(args.data, args.inputs, args.outputs,
                      input_scaler=args.input_scaler,
                      output_scaler=args.output_scaler,
                      classification=args.classification,
                      history=args.history, horizon=args.horizon)

    print(dataset.df)
    
    for subset in dataset.subsets:
        print(subset)
        print(dataset.subsets[subset].df)
        