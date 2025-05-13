#============Modifications to the file============
# function load_pretrained_models: added a condition for loading state with pos_weights
#============Modifications to the file============
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import Transformer
from models.perception import SequentialPerception

class TransformerSudoku(nn.Module):
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
        

    def load_pretrained_models(self, dataset, pos_weight = None):
        perception_path = 'outputs/perception/'+dataset+'/checkpoint_best.pth'
        nn_sol_path = 'outputs/solvernn/'+dataset+'/checkpoint_best.pth'
        if pos_weight == None:
            mask_nn_path = 'outputs/mask/'+dataset+'/checkpoint_best.pth'
        else:
            mask_nn_path = f'outputs/mask/'+dataset+'f/checkpoint_best_{pos_weight}.pth'

        self.perception.load_state_dict(torch.load(perception_path, map_location='cpu'))
        self.nn_solver.load_state_dict(torch.load(nn_sol_path, map_location='cpu'))
        self.mask_nn.load_state_dict(torch.load(mask_nn_path, map_location='cpu'))

    def forward(self, x, nasr = 'rl'):
        
        if nasr == 'pretrained':
            # for eval of pretrained pipeline (NASR w/o RL)
            assert not bool(self.training), f'{nasr} is available only to evaluate. If you want to train it, use the RL pipeline.'
            x0 = self.perception.forward(x)
            x0 = torch.exp(x0)
            a = x0.argmax(dim=2)
            x1 = F.one_hot(a,num_classes=10).to(torch.float32)
            x2 = self.nn_solver.forward(x1)
            b = x2.argmax(dim=2)+1
            x2 = F.one_hot(b,num_classes=10)
            x2 = x2[:,:,1:].to(torch.float32).to(torch.float32)
            x3 = self.mask_nn.forward(x2)
        else:
            # for traning with RL and eval with RL (NASR with RL)
            assert nasr == 'rl', f'{nasr} do not exists, choose between rl and pretrained'
            x0 = self.perception.forward(x)
            x1 = torch.exp(x0)
            x2 = self.nn_solver.forward(x1)
            x2 = F.softmax(x2, dim=2)
            #x2 = F.gumbel_softmax(x2, tau = 1, hard=True, dim=2)
            x3 = self.mask_nn.forward(x2)
            # print(x3[:5])
        return x2, x3


def get_model(block_len=256, **kwargs):
    model = TransformerSudoku(block_len=block_len, **kwargs)
    return model


class TransformerSudokuNoArgmax(nn.Module):
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
        

    def load_pretrained_models(self, dataset, pos_weight = None):
        perception_path = 'outputs/perception/'+dataset+'/checkpoint_best.pth'
        nn_sol_path = 'outputs/solvernn/'+dataset+'/checkpoint_best.pth'
        if pos_weight == None:
            mask_nn_path = 'outputs/mask/'+dataset+'/checkpoint_best.pth'
        else:
            mask_nn_path = f'outputs/mask/'+dataset+'f/checkpoint_best_{pos_weight}.pth'

        self.perception.load_state_dict(torch.load(perception_path, map_location='cpu'))
        self.nn_solver.load_state_dict(torch.load(nn_sol_path, map_location='cpu'))
        self.mask_nn.load_state_dict(torch.load(mask_nn_path, map_location='cpu'))

    def forward(self, x, nasr = 'rl'):
        
        if nasr == 'pretrained':
            # for eval of pretrained pipeline (NASR w/o RL)
            assert not bool(self.training), f'{nasr} is available only to evaluate. If you want to train it, use the RL pipeline.'
            x0 = self.perception.forward(x)
            x0 = torch.exp(x0)
            x1 = x0
            # a = x0.argmax(dim=2)
            # x1 = F.one_hot(a,num_classes=10).to(torch.float32)
            x2 = self.nn_solver.forward(x1)
            # b = x2.argmax(dim=2)+1
            # x2 = F.one_hot(b,num_classes=10)
            # x2 = x2[:,:,1:].to(torch.float32).to(torch.float32)
            x3 = x2
            x3 = self.mask_nn.forward(x2)
        else:
            # for traning with RL and eval with RL (NASR with RL)
            assert nasr == 'rl', f'{nasr} do not exists, choose between rl and pretrained'
            x0 = self.perception.forward(x)
            x1 = torch.exp(x0)
            x2 = self.nn_solver.forward(x1)
            x2 = F.softmax(x2, dim=2)
            #x2 = F.gumbel_softmax(x2, tau = 1, hard=True, dim=2)
            x3 = self.mask_nn.forward(x2)
            # print(x3[:5])
        return x2, x3


def get_no_argmax_model(block_len=256, **kwargs):
    model = TransformerSudokuNoArgmax(block_len=block_len, **kwargs)
    return model




class TransformerSudokuNoPerc(nn.Module):
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
        

    def load_pretrained_models(self, dataset, pos_weight = None):
        perception_path = 'outputs/perception/'+dataset+'/checkpoint_best.pth'
        nn_sol_path = 'outputs/solvernn/'+dataset+'/checkpoint_best.pth'
        if pos_weight == None:
            mask_nn_path = 'outputs/mask/'+dataset+'/checkpoint_best.pth'
        else:
            mask_nn_path = f'outputs/mask/'+dataset+'f/checkpoint_best_{pos_weight}.pth'

        # self.perception.load_state_dict(torch.load(perception_path, map_location='cpu'))
        self.nn_solver.load_state_dict(torch.load(nn_sol_path, map_location='cpu'))
        self.mask_nn.load_state_dict(torch.load(mask_nn_path, map_location='cpu'))

    #takes in the output of perception. x will be fed directly to the solvernn
    def forward(self, x, nasr = 'rl', temperature=1, mask=None):
        
        if nasr == 'pretrained':
            # for eval of pretrained pipeline (NASR w/o RL)
            assert not bool(self.training), f'{nasr} is available only to evaluate. If you want to train it, use the RL pipeline.'
            # x0 = self.perception.forward(x)
            x0 = x
            a = x0.argmax(dim=2)
            x1 = F.one_hot(a,num_classes=10).to(torch.float32)
            x2 = self.nn_solver.forward(x1)
            b = x2.argmax(dim=2)+1
            x2 = F.one_hot(b,num_classes=10)
            x2 = x2[:,:,1:].to(torch.float32).to(torch.float32)
            x3 = self.mask_nn.forward(x2)
        else:
            # for traning with RL and eval with RL (NASR with RL)
            assert nasr == 'rl', f'{nasr} do not exists, choose between rl and pretrained'
            # x0 = self.perception.forward(x)
            # x1 = torch.exp(x)
            solvernn_logits = self.nn_solver.forward(x)
            if temperature == 1:
                x2 = F.softmax(solvernn_logits, dim=2)
            else:
                x2 = self.softmax_with_temperature_3d(solvernn_logits, temperature, mask)
            #x2 = F.gumbel_softmax(x2, tau = 1, hard=True, dim=2)
            x3 = self.mask_nn.forward(x2)
            # print(x3[:5])
        return x2, x3
    
    def softmax_with_temperature_3d(self, solvernn_solutions, temperature=1, mask=None):
        # Scale the logits by the temperature along the 3rd axis (dim=2)
        
        scaled_tensor = solvernn_solutions / temperature
        
        
        if mask != None:
            mask_expanded = mask.unsqueeze(-1)  # Shape: (N, 81, 1)

            scaled_tensor_unmasked = solvernn_solutions  # Temperature = 1 for mask=False

            # Step 3: Combine the scaled matrices based on the mask
            scaled_tensor = torch.where(mask_expanded, scaled_tensor, scaled_tensor_unmasked)
        # print("apply temperature")
        # Apply softmax along the 3rd axis (dim=2)
        return F.softmax(scaled_tensor, dim=2)


def get_no_perc_model(block_len=256, **kwargs):
    model = TransformerSudokuNoPerc(block_len=block_len, **kwargs)
    return model