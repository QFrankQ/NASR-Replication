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

# data_path = 'data/'
# sys.path.append('../data/')

# data = np.load('data/minimal_17/minimal_17-test_inter.npy',allow_pickle=True).item()
data = np.load('data/minimal_17/minimal_17_sol-test.npy',allow_pickle=True).item()

print(data[0])
print(data[0].shape)