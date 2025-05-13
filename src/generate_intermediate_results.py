import torch
import torch.nn.functional as F
import argparse
import argparse
import torch
from models.perception import SequentialPerception, get_PerceptionRL
from datasets import SudokuDataset_Perception
from datasets_inter import SudokuDataset_Solver_Inter
from time import time
import numpy as np
import random
from models.transformer import get_model


def init_parser():
    parser = argparse.ArgumentParser(description='Perception Module for NASR')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    parser.add_argument('--data-type', type=str, default='test',
                        help='data type between [train, val, test]')
    parser.add_argument('--gpu-id', default=0, type=int, help='preferred gpu id')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--module', type=str, default='solvernn',
                        help='data type between [perception, solvernn, mask]')
    args = parser.parse_args()
    return args


def generate_perception_output(args, model, device, test_loader):
    model.eval()
    predictions= []
    targets = []
    with torch.no_grad():
        for (data, target) in test_loader:
            data = data.to(device)
            target = target.to(device)
            preds = model(data)
            preds = torch.exp(preds)
            predictions.append(preds)
            targets.append(target)
            a = preds.size(dim=0)
            b = preds.size(dim=1)
            preds = preds.view(a*b,-1)
            target = target.view(a*b,-1).argmax(dim=1).long()
    predictions = torch.cat(predictions, 0).cpu().numpy()
    targets = torch.cat(targets, 0).cpu().numpy()
    pred = {index: value for index, value in enumerate(predictions)}
    data_out = f'data/{args.data}/{args.data}-{args.data_type}_inter'
    np.save(data_out, pred)
    print("File Saved")
    
def generate_transformer_module_output(args, model, device, test_loader):
    model.eval()
    n = 0
    preds = []
    labels = []
    inputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            pred = model(images)
            
            preds.append(pred)
            labels.append(target)
            inputs.append(images)
            n += images.size(0)


    if args.module == 'mask':
        y_pred = np.round(torch.cat(preds, 0).sigmoid().cpu().numpy())
        y_targ = torch.cat(labels, 0).cpu().numpy()
        # data_out = f'data/{args.data}/{args.data}_{args.data_type}_inter'
        # np.save(data_out, y_pred)

    else:
        assert args.module == 'solvernn'
        y_pred = torch.cat(preds, 0).sigmoid().cpu().numpy()
        y_targ = torch.cat(labels, 0).cpu().numpy()
        input_b = torch.cat(inputs, 0).cpu().numpy()
        # eval_solver(y_pred,y_targ,input_b,dataset_name=args.data)


def main():
    args = init_parser()
    ckpt_path = f'outputs/perception/{args.data}/checkpoint_best.pth'

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.gpu_id if use_cuda else "cpu")
    
    if args.output_of == 'perception':
        dataset = SudokuDataset_Perception(args.data,args.data_type)
    if args.output_of == 'solvernn':
        dataset = SudokuDataset_Solver_Inter(args.data, args.data_type)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    

    if args.module == 'perception':
        model = SequentialPerception()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(device)
        # main loop
        print("Begin generating perception intermediate output")
        generate_perception_output(args, model, device, data_loader)
    else:
        assert args.module in ['solvernn', 'mask']
        num_classes = -1
        in_chans = -1
        if args.module == 'solvernn':
            in_chans = 10
            num_classes = 9
        else:
            assert args.module == 'mask'
            in_chans = 9
            num_classes = 1
        model = get_model(block_len=args.block_len, in_chans = in_chans, num_classes = num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(args.gpu_id)
        
        # main loop
        print("Begin generating solvernn intermediate output")
        generate_perception_output(args, model, device, data_loader)
    


if __name__ == '__main__':

    main()