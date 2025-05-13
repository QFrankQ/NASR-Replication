import os
import json
import argparse
from time import time
import math
import random
import torch
import torch.nn as nn
from models.transformer import get_model
from datasets import SudokuDataset_Mask,SudokuDataset_Solver
import numpy as np
from utils.utils import print_loss_graph_from_file


def init_parser():
    parser = argparse.ArgumentParser(description='Solver-NN and Mask-Predictor Module for NASR')
    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--module', type=str, default='mask',
                        help='module name between mask (mask-predictor) or solvernn')
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--pos-weights', default=None, type=float,
                        help='ratio neg/pos examples')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between [big_kaggle, minimal_17, multiple_sol, satnet]')
    # Optimization hyperparams
    parser.add_argument('--epochs', default=2000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--warmup', default=10, type=int, 
                        help='number of warmup epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)', dest='batch_size')
    parser.add_argument('--lr', default=0.0005, type=float, 
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=3e-2, type=float, 
                        help='weight decay (default: 3e-2)')
    parser.add_argument('--clip-grad-norm', default=0., type=float, 
                        help='gradient norm clipping (default: 0 (disabled))')
    parser.add_argument('--disable-cos', action='store_true',
                        help='disable cosine lr schedule')
    #mod here for file name
    parser.add_argument('--file-name', type=str,
                        help='output file name')
    parser.add_argument('--constraint-loss', type=str,
                        help='constraint specification for loss [cosine, kl]')
    parser.add_argument('--reg-scale', type=float, default=0,
                        help='scale for constraint loss')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    
    assert args.module in ['mask', 'solvernn'], 'error module name, choose between solvernn and mask'

    if args.module == 'mask':
        train_dataset = SudokuDataset_Mask(args.data,'-train')
        val_dataset = SudokuDataset_Mask(args.data,'-valid')
    elif args.module == 'solvernn':
        train_dataset = SudokuDataset_Solver(args.data,'-train')
        val_dataset = SudokuDataset_Solver(args.data,'-valid')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)

    # Model
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
    model.to(args.gpu_id)

    # Loss
    criterion = nn.BCEWithLogitsLoss()
    if args.module == 'mask' and not args.pos_weights is None : 
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weights))
 
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Main loop
    print("Beginning training")
    print(f"constraint loss: {args.constraint_loss}, reg_scale: {args.reg_scale}, lr: {args.lr}")
    args.module = os.path.join('outputs', args.module+'/'+args.data)
    os.makedirs(args.module, exist_ok=True)
    best_loss = 100.
    time_begin = time()
    with open(f"{args.module}/log.txt", 'w'): pass
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        loss = validate(val_loader, model, criterion, args, epoch=epoch, time_begin=time_begin)

        if loss < best_loss:
            best_loss = loss
            #mod here for file name
            torch.save(model.state_dict(), f'{args.module}/checkpoint_best_{args.file_name}_{args.reg_scale}.pth')

        stats = {'epoch': epoch, 'lr': lr,
                 'train_loss': train_loss, 'val_loss': loss, 'best_loss': best_loss}
        with open(f"{args.module}/log.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")

    total_mins = (time() - time_begin) / 60
    print(f'[{args.module}] finished in {total_mins:.2f} minutes, '
          f'best loss: {best_loss:.6f}, '
          f'final loss: {loss:.6f}')
    #mod here for file name
    torch.save(model.state_dict(), f'{args.module}/checkpoint_last_{args.file_name}_{args.reg_scale}.pth')
    print_loss_graph_from_file(f"{args.module}/log.txt",f"{args.module}/loss_{args.file_name}_{args.reg_scale}.png")
    

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    preRegLoss_val = 0
    loss_val = 0
    n = 0
    # print("no standard loss")
    for i, (images, target) in enumerate(train_loader):
        images = images.to(args.gpu_id)
        target = target.to(args.gpu_id)
        
        # mod here to add a reg term to the loss - cosine similarity or kl
        pred = model(images)
        # pred has shape (N, pos, value) = (N, 81, 9)
        # i.e. in board i, position j takes value k with unnormalized probability = pred[i,j,k]
        reg = 0
        # mask has (81,81) and mask[i,j] = 1 iff position[i] is in the same row, col, or block as position[j]. Else = 0
        mask = torch.load('src/mask.pt', weights_only=False)
        mask = mask.cuda()
        for nPred in range(pred.shape[0]):
            boardN = pred[nPred] # boardN has shape (pos,value) = (81,9)
            boardN = nn.functional.softmax(boardN, dim=1) # softmax to normalize the probabilities before performing kl
            if args.constraint_loss == 'cosine':
                matrix1 = boardN.unsqueeze(1)  # Shape: (81, 1, 9)
                matrix2 = boardN.unsqueeze(0)  # Shape: (1, 81, 9)
                # cosine_similarity produces a board with shape (81,81) where board[i,j] measures the likelihood of position[i] and position[j] taking the same value
                conflictLikelihood = nn.functional.cosine_similarity(matrix1, matrix2, dim=-1)
                # print(conflictLikelihood.size())
            elif args.constraint_loss == 'js':
                # jensenShannon produces a board with shape (81,81) where board[i,j] measure the likelihood of position[i] and position[j] taking the same value
                boardNbroad1 = boardN.unsqueeze(1)  # Shape: (81, 1, 9)
                boardNbroad0 = torch.clip(boardN.unsqueeze(0), min=1e-5)  # Shape: (1, 81, 9)
                kl1 = nn.functional.kl_div(boardNbroad1.log(), boardNbroad0, reduction='none').sum(dim=2)
                kl2 = nn.functional.kl_div(boardNbroad0.log(), boardNbroad1, reduction='none').sum(dim=2)
                jensenShannon = 0.5 * (kl1 + kl2)
                conflictLikelihood = -1 * jensenShannon 
            elif args.constraint_loss == 'kl':
                # kl_div produces a board with shape (81,81) where board[i,j] measure the likelihood of position[i] and position[j] taking the same value
                boardNbroad1 = boardN.unsqueeze(1)  # Shape: (81, 1, 9)
                boardNbroad0 = torch.clip(boardN.unsqueeze(0), min=1e-5)  # Shape: (1, 81, 9)
                kl = nn.functional.kl_div(boardNbroad1.log(), boardNbroad0, reduction='none').sum(dim=2)
                # print(kl.size())
                conflictLikelihood = -1 * kl  
            else: #standard
                conflictLikelihood = 0.0
            # multiplying by mask sets positions that cannot conflict to 0 and leaves the others unchanged
            # print(mask.size())
            regN = torch.sum(conflictLikelihood * mask)
            # print(type(regN))
            reg += regN
            # print(type(reg))
        # scale down reg by scale factor from args
        reg *= args.reg_scale
        reg /= pred.shape[0]
        loss = criterion(pred, target)
        # # mod here to add reg to loss and create total loss
        total_loss = loss + reg
        
        # total_loss = reg
        n += images.size(0)
        loss_val += float(total_loss.item() * images.size(0))
        preRegLoss_val += float(loss.item() * images.size(0))
        
        optimizer.zero_grad()
        # mod here to backprop on total loss instead of loss
        total_loss.backward()

        if args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

        optimizer.step()

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss = (loss_val / n)
            avgPreReg_loss = (preRegLoss_val / n)
            print(f'preRegLoss: [{args.module}][Epoch {epoch}][Train][{i}] \t AvgLoss: {avgPreReg_loss:.4f}')
            print(f'regLoss:    [{args.module}][Epoch {epoch}][Train][{i}] \t AvgLoss: {avg_loss:.4f}')

    avg_loss = (loss_val / n)
    return avg_loss 


def validate(val_loader, model, criterion, args, epoch=None, time_begin=None):
    model.eval()
    loss_val = 0
    n = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.gpu_id)
            target = target.to(args.gpu_id)

            pred = model(images) #torch.Size([100, 81, 9])
            loss = criterion(pred, target)

            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss = (loss_val / n)
                print(f'[{args.module}][Epoch {epoch}][Val][{i}] \t AvgLoss: {avg_loss:.4f}')

    avg_loss = (loss_val / n)
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f'[{args.module}][Epoch {epoch}] \t \t AvgLoss {avg_loss:.4f} \t \t Time: {total_mins:.2f}')
    print(f"constraint loss: {args.constraint_loss}, reg_scale: {args.reg_scale}, lr: {args.lr}")
    return avg_loss


if __name__ == '__main__':
    main()
    
