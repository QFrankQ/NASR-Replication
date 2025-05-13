from ctypes import addressof
from re import A
from matplotlib import transforms
from torchvision import transforms as T
from PIL import Image
import numpy as np
from time import time
import torch
from torch.utils import data
import torch.nn.functional as F
import h5py
from torch.utils.data import TensorDataset
import pickle
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from tqdm import tqdm
from sudoku_data import process_inputs
import sys 

data_path = 'data/'
sys.path.append('../data/')


#y may or may not be one-hot encoded 
class SudokuDataset_Solver_Inter(data.Dataset):
    # x 9x9x10
    # y 9x9x9
    def __init__(self, filename, data_type, nasr='rl'):
        assert filename in ['big_kaggle', 'minimal_17', 'multiple_sol', 'satnet'], 'error in the dataset name. Available datasets are big_kaggle, minimal_17, multiple_sol, and satnet'
        assert data_type in ['-test','-train','-valid'], 'error, types allowed are -test, -train, -valid.'
        if nasr == 'rl':
            self.data = np.load(data_path+filename+'/'+filename + data_type + '_interRL.npy',allow_pickle=True).item()
        else:
            assert nasr == 'pretrained', f'{nasr} not supported, choose between pretrained and rl'
            self.data = np.load(data_path+filename+'/'+filename + data_type + '_inter.npy',allow_pickle=True).item()
        self.labels = np.load(data_path+filename+'/'+filename+'_sol'+ data_type +'.npy',allow_pickle=True).item()
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        xt = torch.from_numpy(self.data[idx])
        yt = torch.from_numpy(self.labels[idx].reshape(81))
        yt = F.one_hot(yt,num_classes=10)
        # yt = yt[:,1:]
        return xt.to(torch.float32), yt.to(torch.float32)


if __name__ == '__main__':
    dataset = SudokuDataset_Solver_Inter('minimal_17', '-test')
    print(dataset[0])