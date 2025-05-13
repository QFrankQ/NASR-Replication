import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.transformer import Transformer
from functools import partial

class DigitRecognition(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self):
        super(DigitRecognition, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 10, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1440, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output_ls = F.log_softmax(x, dim=1)
        return output_ls

class SequentialPerception(nn.Module):
    '''
    Convolutional neural network for MNIST digit recognition. From:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    '''
    def __init__(self):
        super(SequentialPerception, self).__init__()
        self.digit_rec = DigitRecognition()
        

    def forward(self, x):
        x = x.flatten(start_dim = 0,end_dim=1)
        digit_pred = self.digit_rec(x)
        batch_size = int(len(x)/81)
        puzzles = digit_pred.view(batch_size,81,10)
        return puzzles

class PerceptionRL(nn.Module):
    def __init__(self, block_len=256, **kwargs):
        super().__init__()
        self.saved_log_probs = []
        self.rewards = []
        self.perception = SequentialPerception()
        self.nn_solver = Transformer(in_chans=10, num_classes=9,    
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        self.mask_nn = Transformer(in_chans=9, num_classes=1,
                        block_len=block_len, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    def forward(self, x):
        
        # for traning with RL and eval with RL (NASR with RL)
        x0 = self.perception.forward(x)
        return x0


def get_PerceptionRL(block_len=256, **kwargs):
    model = PerceptionRL(block_len=block_len, **kwargs)
    return model
