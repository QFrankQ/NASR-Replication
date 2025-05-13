import os
import json
import argparse
from time import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from models.transformer_sudoku import get_model, get_no_argmax_model, get_no_perc_model
from datasets import SudokuDataset_RL
from datasets_inter import SudokuDataset_Solver_Inter
import random
from rl_train_sudoku import compute_reward
from sudoku_solver.board import Board
# try:
#     from pyswip import Prolog
# except Exception:
#     print('-->> Prolog not installed')
from torch.distributions.bernoulli import Bernoulli
from sudoku_solver.board import check_input_board,check_consistency_board
from models.perception import SequentialPerception
from tqdm import tqdm
from utils.utils import retrieve_hints_from_solution
# from scipy.stats import entropy
import matplotlib

def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

    # General args
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='log frequency (by iteration)')
    parser.add_argument('--batch-size', default=100, type=int,
                        help='Batch size')
    parser.add_argument('--nasr', type=str, default='rl',
                        help='choice of nasr with nasr_rl or pretrained (without rl)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str,
                        help='Json file of noise setting (dict)')
    parser.add_argument('--solver', type=str, default='prolog',
                        help='symbolic solver to use. available options prolog and backtrack')
    parser.add_argument('--transform-data', type=str, default=None,
                        help='noise type to add to the images of sudoku digits. available options are blur and rotation')
    parser.add_argument('--transform-data-param', default=0.0, type=float,
                        help='noise value. angle degrees for rotation and sigma for blur.')
    parser.add_argument('--modified-module', default=False, type=bool,
                        help='load modified model')
    parser.add_argument('--analysis', default=0, type=int,
                        help='single temperature evaluation or evalaute with a list of temperature')
    parser.add_argument('--performance-mask', default=None, type=str,
                        help='type of performance mask to be applied during softmax, possible values:')
    # hint_cells_indices             # sol_cells_indices
    # correct_sol_cells_indices      # error_sol_cells_indices
    # masked_error_sol_cells_indices # unmask_error_sol_cells_indices
    
    # Model args
    parser.add_argument('--block-len', default=81, type=int,
                        help='board size')
    parser.add_argument('--code-rate', default=2, type=int,
                        help='Code rate')
    parser.add_argument('--pos-weights', default=None, type=float,
                    help='ratio neg/pos examples')
    parser.add_argument('--temp', default=1, type=float,
                        help='temperature for softmaxing solvernn output')
    return parser


# def final_output(model,ground_truth_sol,solution_boards,masking_boards,args):
#     ground_truth_boards = torch.argmax(ground_truth_sol,dim=2)
#     solution_boards_new = torch.argmax(solution_boards,dim=2)+1
#     # using sigmoid_round
#     masking_prob = masking_boards.sigmoid()
#     b = Bernoulli(masking_prob)
#     mask_boards = torch.round(masking_prob)
#     model.saved_log_probs = b.log_prob(mask_boards)
#     cleaned_boards = np.multiply(solution_boards_new.cpu(),mask_boards.cpu())
#     final_boards = []
#     if args.solver == "prolog":
#         # prolog_instance = Prolog()
#         # prolog_instance.consult("src/sudoku_solver/sudoku_prolog.pl")
#         pass
#     for i in range(len(cleaned_boards)):
#         board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int())
#         time_begin = time()
#         try:            
#             if args.solver == "prolog":
#                 solver_success = board_to_solver.solve(solver ='prolog',prolog_instance = prolog_instance)
#             else:
#                 solver_success = board_to_solver.solve(solver ='backtrack')
#         except StopIteration:
#             solver_success = False
#         time_solver = (time() - time_begin)
#         final_solution = board_to_solver.board.reshape(81,)
#         if not solver_success:
#             final_solution = solution_boards_new[i].cpu()
#         reward = compute_reward(solution_boards_new[i].cpu(),final_solution,ground_truth_boards[i])
#         model.rewards.append(reward)
#         final_boards.append(final_solution)
#     return final_boards, time_solver

def solvernn_entropy(solution_boards):
    assert solution_boards.shape[1:] == (81, 9), "Input shape must be (N, 81, 9)"

    # Normalize along the last axis
    # Add a small constant to avoid log(0) (numerical stability)
    epsilon = 1e-7
    probs = torch.clamp(solution_boards, min=epsilon, max=1.0 - epsilon)

    # Compute entropy along the last axis
    entropy_matrix = -torch.sum(probs * torch.log(probs), dim=-1)

    # Return the average entropy
    return torch.mean(entropy_matrix).item()

def mask_entropy(probs):
    # Ensure that probabilities are in the range [epsilon, 1-epsilon] to avoid log(0)
    epsilon = 1e-7
    probs = torch.clamp(probs, min=epsilon, max=1.0 - epsilon)
    # Calculate the entropy for each vector in the batch
    
    entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))
    # Return the mean entropy across the batch
    return entropy.mean().item()
    

def validate(val_loader, model, args, epoch=None, time_begin=None):
    model.eval()
    loss_value = 0
    n = 0
    total_solvernn_entropy = 0 
    total_mask_entropy = 0
    preds = []
    labels = []
    inputs = []
    solutions = []
    mask = []
    masking_probs = []
    eps = np.finfo(np.float32).eps.item()
    time_begin = time()
    performance_dict = {}
    
    with torch.no_grad():
        for i, (perception_boards, target) in enumerate(val_loader):
            
            perception_boards = perception_boards.to(args.gpu_id)
            target = target.to(args.gpu_id)
            solution_boards, masking_boards = model(perception_boards, nasr=args.nasr, temperature =args.temp)
            
            # pred_final, _ = final_output(model,target,solution_boards,masking_boards,args)
            if args.performance_mask != None:
                y_targ = target.cpu().numpy()
                
                mask_b = torch.round(masking_boards.sigmoid()).cpu().numpy()
                
                pred_boards_b = solution_boards.sigmoid().cpu().numpy()
                y_pred_b = []
                for idx in range(len(pred_boards_b)):
                    r_pred_board = np.zeros((81), dtype=int)
                    pred_board = pred_boards_b[idx]
                    for k in range(81):
                        pred_cell = pred_board[k].argmax()+1
                        r_pred_board[k] = float(pred_cell)
                    y_pred_b.append(r_pred_board)
                y_pred_b = np.array(y_pred_b)
                print("Getting performance masks")
                performance_masks_dict = get_performance_masks(y_pred_b,mask_b,y_targ,args)
                performance_mask = torch.tensor(performance_masks_dict[args.performance_mask]).to(args.gpu_id)
                solution_boards, masking_boards = model(perception_boards, nasr=args.nasr, temperature =args.temp, mask=performance_mask)
            masking_prob = masking_boards.sigmoid()
            masking_probs.append(masking_prob)
            
            mask.append(torch.round(masking_prob))
            # preds.append(pred_final)
            labels.append(target)
            # inputs.append(images)
            inputs.append(perception_boards)
            solutions.append(solution_boards)
        
            # policy_loss = []
            # for reward, log_prob in zip(rewards, model.saved_log_probs):
            #     policy_loss.append(-log_prob*reward)
            # policy_loss = (torch.cat(policy_loss)).sum()

            n += perception_boards.size(0)
            total_solvernn_entropy += solvernn_entropy(solution_boards) * perception_boards.size(0)
            total_mask_entropy += mask_entropy(masking_prob) * perception_boards.size(0)
            
            # loss_value += float(policy_loss.item() * images.size(0))
            # model.rewards = []
            # model.saved_log_probs = []
            torch.cuda.empty_cache()

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_solvernn_entropy = (total_solvernn_entropy / n)
                avg_mask_entropy = (total_mask_entropy/n)
                print(f'[Test][{i}/{len(val_loader)}] \t AvgSolvernnEntropy: {avg_solvernn_entropy:.4f} \t AvgMaskEntropy: {avg_mask_entropy:.4f}')
            if i >49:
                break
            
    avg_solvernn_entropy = (total_solvernn_entropy/ n)
    avg_mask_entropy = (total_mask_entropy/n)
    avg_solvernn_confidence = -avg_solvernn_entropy
    avg_mask_confidence = -avg_mask_entropy
    total_mins = -1 if time_begin is None else (time() - time_begin) / 60
    print(f"Temperature for normalizing solvernn output logits: {args.temp}")
    print(f'AvgSolvernnConfidence {avg_solvernn_confidence:.4f} \t \t AvgMaskConfidence {avg_mask_confidence:.4f} \t \t Time: {total_mins:.2f}')
    performance_dict['avg_solvernn_confidence'] = avg_solvernn_confidence
    performance_dict['avg_mask_confidence'] = avg_mask_confidence
    performance_dict['temp'] = args.temp
    
    concat_solutions = torch.cat(solutions,0)
    concat_masking_probs = torch.cat(masking_probs,0)
    pred_boards_b = concat_solutions.sigmoid().cpu().numpy()
    
    y_pred_b = []
    for idx in range(len(pred_boards_b)):
        r_pred_board = np.zeros((81), dtype=int)
        pred_board = pred_boards_b[idx]
        for k in range(81):
            pred_cell = pred_board[k].argmax()+1
            r_pred_board[k] = float(pred_cell)
        y_pred_b.append(r_pred_board)
    y_pred = []
    # for j in preds:
    #     for i in j:
    #         y_pred.append(i) 
    y_targ = torch.cat(labels, 0).cpu().numpy()
    input_b = torch.cat(inputs, 0).cpu().numpy()
    mask_b = torch.cat(mask, 0).cpu().numpy()
    
    # print("-------------------------------")
    # print("Improvement:")
    # print("-----------")
    # eval_improvement(y_pred,y_pred_b,y_targ,args.data)
    
    print("-------------------------------")
      
    print("-------------------------------")
    print("Eval intuition")
    performance_dict = intuition_eval(input_b,y_pred,y_pred_b,mask_b,y_targ,args, performance_dict)
    print("-------------------------------")
    print("Confidence evaluation")
    performance_dict = confidence_eval(concat_solutions, concat_masking_probs, y_pred_b,mask_b,y_targ,args, performance_dict)
    return performance_dict

def get_performance_masks(y_pred_b,mask_b,y_targ,args):
    hint_cells_indices = []
    sol_cells_indices = []
    correct_sol_cells_indices = []
    error_sol_cells_indices = []
    masked_error_sol_cells_indices = []
    unmask_error_sol_cells_indices = []
    performance_masks = {}
    for i in range(len(y_pred_b)):
        # print(f"y_pred_b size", len(solution_boards))
        # perception_board = perception_boards[i].reshape(81)
        neuro_solver_board = np.array(y_pred_b[i]).reshape(81)
        mask_board = np.array(mask_b[i]).astype(int)
        # pipeline_board = np.array(y_pred[i]).reshape(81)
        gt_board = np.array(y_targ[i]).reshape((81,10))
        gt_board = gt_board.argmax(axis=1)

        input_board = retrieve_hints_from_solution(gt_board,args.data).reshape(81)
        
        hint_cells_idx = input_board==0
        sol_cells_idx = input_board!=0
        hint_cells_indices.append(hint_cells_idx)
        sol_cells_indices.append(sol_cells_idx)
        #index of correct and error cells
        correct_cells = neuro_solver_board == gt_board
        error_cells = neuro_solver_board != gt_board
                
        #index of correct and error solution cells
        correct_sol_cells_idx = correct_cells & sol_cells_idx
        error_sol_cells_idx = error_cells & sol_cells_idx
        correct_sol_cells_indices.append(correct_sol_cells_idx)
        error_sol_cells_indices.append(error_sol_cells_idx)
        
        #get index of masked and unmasked cells
        masked_cells = mask_board == 0
        unmasked_cells = mask_board !=0
        
        #get index of caught and missed error solution cells
        masked_error_sol_cells_idx = error_sol_cells_idx & masked_cells
        unmask_error_sol_cells_idx = error_sol_cells_idx & unmasked_cells
        masked_error_sol_cells_indices.append(masked_error_sol_cells_idx)
        unmask_error_sol_cells_indices.append(unmask_error_sol_cells_idx)
    # print(f"hint_cells_indices len:",len(hint_cells_indices))
    performance_masks['hint_cells_indices'] = np.array(hint_cells_indices)
    performance_masks['sol_cells_indices'] = np.array(sol_cells_indices)
    performance_masks['correct_sol_cells_indices'] = np.array(correct_sol_cells_indices)
    performance_masks['error_sol_cells_indices'] = np.array(error_sol_cells_indices)
    performance_masks['masked_error_sol_cells_indices'] = np.array(masked_error_sol_cells_indices)
    performance_masks['unmask_error_sol_cells_indices'] = np.array(unmask_error_sol_cells_indices)
    return performance_masks
    
    
def confidence_eval(solution_boards, masking_prob, y_pred_b,mask_b,y_targ,args, performance_dict):
    solvernn_hint_cf = []
    solvernn_sol_cf = [] 
    solvernn_correct_sol_cf = []
    solvernn_error_sol_cf = []
    solvernn_catch_error_sol_cf = []
    solvernn_miss_error_sol_cf = []
    mask_hint_cf = []
    mask_sol_cf = []
    mask_correct_sol_cf = []
    mask_error_sol_cf = []
    mask_catch_error_sol_cf = []
    mask_miss_error_sol_cf = []
    
    
    def compute_solvernn_confidence(dist): #takes in matrix of (n, 9)
        return torch.sum(dist * torch.log(dist), dim=1).mean().item()
    
    def compute_mask_confidence(probs): #takes in a vector of entries probability p 
        return (probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).mean().item()
    
    for i in tqdm(range(len(y_pred_b))):
        # perception_board = perception_boards[i].reshape(81)
        neuro_solver_board = np.array(y_pred_b[i]).reshape(81)
        mask_board = np.array(mask_b[i]).astype(int)
        # pipeline_board = np.array(y_pred[i]).reshape(81)
        gt_board = np.array(y_targ[i]).reshape((81,10))
        gt_board = gt_board.argmax(axis=1)

        input_board = retrieve_hints_from_solution(gt_board,args.data).reshape(81)
        
        hint_cells_idx = input_board==0
        sol_cells_idx = input_board!=0
        epsilon = 1e-7
        solution_board_i = torch.clamp(solution_boards[i], min=epsilon, max=1.0)
        masking_prob_i = torch.clamp(masking_prob[i], min=epsilon, max=1.0)
        
        
        #probability distribution of solvernn and mask for both hint and sol cells
        solvernn_hint_distribution = solution_board_i[hint_cells_idx]
        solvernn_sol_distribution = solution_board_i[sol_cells_idx]
        mask_hint_prob_i = masking_prob_i[hint_cells_idx]
        masking_sol_prob_i = masking_prob_i[sol_cells_idx]
        
        #compute solvernn and mask's respective confidence in hint and sol cells
        solvernn_hint_cf.append(compute_solvernn_confidence(solvernn_hint_distribution))
        solvernn_sol_cf.append(compute_solvernn_confidence(solvernn_sol_distribution))
        mask_hint_cf.append(compute_mask_confidence(mask_hint_prob_i))
        mask_sol_cf.append(compute_mask_confidence(masking_sol_prob_i))
        
        #index of correct and error cells
        correct_cells = neuro_solver_board == gt_board
        error_cells = neuro_solver_board != gt_board
                
        #index of correct and error solution cells
        correct_sol_cells_idx = correct_cells & sol_cells_idx
        error_sol_cells_idx = error_cells & sol_cells_idx
        
        #probability distribution of solvernn & mask for correct and error solution cells
        solvernn_correct_sol_dist = solution_board_i[correct_sol_cells_idx]
        solvernn_error_sol_dist = solution_board_i[error_sol_cells_idx]
        mask_correct_sol_prob = masking_prob_i[correct_sol_cells_idx]
        mask_error_sol_prob = masking_prob_i[error_sol_cells_idx]
        
        #check the existence of correct and error solution cells then append solvernn and masks confidence
        if np.any(correct_sol_cells_idx):
            solvernn_correct_sol_cf.append(compute_solvernn_confidence(solvernn_correct_sol_dist))
            mask_correct_sol_cf.append(compute_mask_confidence(mask_correct_sol_prob))
        if np.any(error_sol_cells_idx):
            solvernn_error_sol_cf.append(compute_solvernn_confidence(solvernn_error_sol_dist))
            mask_error_sol_cf.append(compute_mask_confidence(mask_error_sol_prob))
        
        #get index of masked and unmasked cells
        masked_cells = mask_board == 0
        unmasked_cells = mask_board !=0
        
        #get index of caught and missed error solution cells
        masked_error_sol_cells_idx = error_sol_cells_idx & masked_cells
        unmask_error_sol_cells_idx = error_sol_cells_idx & unmasked_cells
        
        #get distribution
        solvernn_caught_error_sol_cells_dist = solution_board_i[masked_error_sol_cells_idx]
        solvernn_miss_error_sol_cells_dist = solution_board_i[unmask_error_sol_cells_idx]
        mask_caught_error_sol_cells_prob = masking_prob_i[masked_error_sol_cells_idx]
        mask_miss_error_sol_cells_prob = masking_prob_i[unmask_error_sol_cells_idx]
        
        if np.any(masked_error_sol_cells_idx):
            solvernn_catch_error_sol_cf.append(compute_solvernn_confidence(solvernn_caught_error_sol_cells_dist))
            mask_catch_error_sol_cf.append(compute_mask_confidence(mask_caught_error_sol_cells_prob))
        if np.any(unmask_error_sol_cells_idx):
            solvernn_miss_error_sol_cf.append(compute_solvernn_confidence(solvernn_miss_error_sol_cells_dist))
            mask_miss_error_sol_cf.append(compute_mask_confidence(mask_miss_error_sol_cells_prob))
            
    solvernn_hint_cf = np.mean(solvernn_hint_cf)
    solvernn_sol_cf = np.mean(solvernn_sol_cf) 
    solvernn_correct_sol_cf = np.mean(solvernn_correct_sol_cf)
    solvernn_error_sol_cf = np.mean(solvernn_error_sol_cf)
    solvernn_catch_error_sol_cf = np.mean(solvernn_catch_error_sol_cf)
    solvernn_miss_error_sol_cf = np.mean(solvernn_miss_error_sol_cf)
    mask_hint_cf = np.mean(mask_hint_cf)
    mask_sol_cf = np.mean(mask_sol_cf) 
    mask_correct_sol_cf = np.mean(mask_correct_sol_cf) 
    mask_error_sol_cf = np.mean(mask_error_sol_cf)
    mask_catch_error_sol_cf = np.mean(mask_catch_error_sol_cf) 
    mask_miss_error_sol_cf = np.mean(mask_miss_error_sol_cf)  

    print(f"Average model confidence in hint cells: SolverNN: {solvernn_hint_cf}, Mask: {mask_hint_cf}")
    print(f"Average model confidence in sol cells: SolverNN: {solvernn_sol_cf}, Mask: {mask_sol_cf}")
    print(f"Average model confidence in correct sol cells: SolverNN: {solvernn_correct_sol_cf}, Mask: {mask_correct_sol_cf}")
    print(f"Average model confidence in error sol cells: SolverNN: {solvernn_error_sol_cf}, Mask: {mask_error_sol_cf}")
    print(f"Average model confidence in masked error sol cells: SolverNN: {solvernn_catch_error_sol_cf}, Mask: {mask_catch_error_sol_cf}")
    print(f"Average model confidence in missed error sol cells: SolverNN: {solvernn_miss_error_sol_cf}, Mask: {mask_miss_error_sol_cf}")
    performance_dict['solvernn_hint_cf'] = solvernn_hint_cf
    performance_dict['solvernn_sol_cf'] = solvernn_sol_cf 
    performance_dict['solvernn_correct_sol_cf'] = solvernn_correct_sol_cf
    performance_dict['solvernn_error_sol_cf'] = solvernn_error_sol_cf
    performance_dict['solvernn_catch_error_sol_cf'] = solvernn_catch_error_sol_cf
    performance_dict['solvernn_miss_error_sol_cf'] = solvernn_miss_error_sol_cf
    performance_dict['mask_hint_cf'] = mask_hint_cf
    performance_dict['mask_sol_cf'] = mask_sol_cf
    performance_dict['mask_correct_sol_cf'] = mask_correct_sol_cf
    performance_dict['mask_error_sol_cf'] = mask_error_sol_cf
    performance_dict['mask_catch_error_sol_cf'] = mask_catch_error_sol_cf
    performance_dict['mask_miss_error_sol_cf'] = mask_miss_error_sol_cf
    return performance_dict


def intuition_eval(input_boards,y_pred,y_pred_b,mask_b,y_targ,args, performance_dict):
    # perception_path = 'outputs/perception/'+args.data+'/checkpoint_best.pth'
    # perception = SequentialPerception()
    # perception.to(args.gpu_id)
    # perception.load_state_dict(torch.load(perception_path, map_location='cpu')) 
    # input_boards = (torch.from_numpy(input_boards)).to(torch.float32).to(args.gpu_id)
    # perception_boards = None
    # with torch.no_grad():
    #     for i in input_boards:
    #         result_tensor = perception(i.unsqueeze(dim=0))
    #         if perception_boards is None:
    #             perception_boards = result_tensor
    #         else:
    #             perception_boards = torch.cat((perception_boards,result_tensor))
    # perception_boards = input_boards.clone().detach()
    # perception_boards = torch.exp(perception_boards)
    # perception_boards = perception_boards.argmax(dim=2)
    # perception_boards = perception_boards.cpu().detach().numpy()
    n_error_p = []
    n_errors_p_corrected_by_ns = []
    n_error_perception_masked = []
    n_error_ns = []
    perc_error_ns_i = []
    perc_error_ns_s = []
    perc_error_ns_masked_i = []
    perc_error_ns_masked_s = []
    perc_correct_ns_preserved_s = []
    perc_correct_ns_preserved_i = []
    perc_error_ns_corrected_i = []
    perc_error_ns_corrected_s = []
    perc_cells_masked = []
    n_hints = []
    n_solutions = []
    
    for i in tqdm(range(len(y_pred_b))):
        # perception_board = perception_boards[i].reshape(81)
        neuro_solver_board = np.array(y_pred_b[i]).reshape(81)
        mask_board = np.array(mask_b[i]).astype(int)
        # pipeline_board = np.array(y_pred[i]).reshape(81)
        gt_board = np.array(y_targ[i]).reshape((81,10))
        gt_board = gt_board.argmax(axis=1)

        input_board = retrieve_hints_from_solution(gt_board,args.data).reshape(81)
        
        nep = 0 #number error perception #added
        nepc = 0 # number error perception corrected
        nep_m = 0 # number error perception corrected by the mask
        nep_cnp = 0 # number error perception corrected by solverNN

        nens = 0 #number error neuro solver
        ncns_s = 0 #number correct solutions from neuro solver
        ncns_i = 0 #number correct inputs from neuro solver
        nens_s = 0 # number error neuro solver, in the solutions
        nens_i = 0 # number error neuro solver, in the inputs
        ncns_p_s = 0 #number of correct solutions being preserved
        ncns_p_i = 0 #number of correct inputs being preserved
        nens_m_s = 0 # number error solverNN, in the solutions, that are found by the mask
        nens_m_i = 0 # number error solverNN, in the inputs, that are found by the mask
        nens_c_s = 0 # number error solverNN, in the solutions, that have been corrected by the pipeline
        nens_c_i = 0 # number error solverNN, in the inputs, that have been corrected by the pipeline

        n_sol_cells = np.sum(input_board==0)
        n_hint_cells = 81 - n_sol_cells
        n_solutions.append(n_sol_cells)
        n_hints.append(n_hint_cells)
        for j in range(81):
        
            # if input_board[j]!= 0 and input_board[j]!=perception_board[j]: # error perception
            #     nep +=1
            #     if pipeline_board[j] == gt_board[j]: # has been corrected
            #         nepc+=1
            #         if neuro_solver_board[j] == gt_board[j]:
            #             nep_cnp += 1
            #         else:
            #             if mask_board[j] == 0:
            #                 nep_m += 1
            
            # assert nepc == nep_cnp + nep_m

            if neuro_solver_board[j]!=gt_board[j]: # error solverNN 
                nens +=1
                if input_board[j]== 0: # error in the solutions
                    nens_s+=1
                    # if pipeline_board[j] == gt_board[j]: # has been corrected
                    #     nens_c_s+=1
                    if mask_board[j] == 0:
                        nens_m_s+=1
                else: # error in the input
                    nens_i+=1
                    # if pipeline_board[j] == gt_board[j]: # has been corrected
                    #     nens_c_i+=1
                    if mask_board[j] == 0:
                        nens_m_i+=1
            else: #correct solverNN
                if input_board[j] == 0:  #correct solution solverNN
                    ncns_s += 1
                    if mask_board[j] == 1:
                        ncns_p_s += 1
                else:
                    ncns_i += 1
                    if mask_board[j] == 1:
                        ncns_p_i += 1

        perc_cells_masked.append(np.sum(mask_board)/81)

        if ncns_s != 0: #added
            perc_correct_ns_preserved_s.append(ncns_p_s/ncns_s)
        if ncns_i != 0:
            perc_correct_ns_preserved_i.append(ncns_p_i/ncns_i)
        if nep !=0: #added
            n_error_p.append(nep/81)
        if nepc!=0:
            n_errors_p_corrected_by_ns.append(nep_cnp/nepc)
            n_error_perception_masked.append(nep_m/nepc)
        if nens !=0:
            n_error_ns.append(nens)#added
        if nens_i!=0:
            perc_error_ns_masked_i.append(nens_m_i/nens_i)
            perc_error_ns_corrected_i.append(nens_c_i/nens_i)
            perc_error_ns_i.append(nens_i/n_hint_cells)#added
        if nens_s!=0:
            perc_error_ns_masked_s.append(nens_m_s/nens_s)
            perc_error_ns_corrected_s.append(nens_c_s/nens_s)
            perc_error_ns_s.append(nens_s/n_sol_cells)#added
    print("evaluated for-loop")
    # print(f"Num errors of the perception (avg): {statistics.mean(n_error_p)}%)")#added
    # print(f"Num errors of the perception corrected by the SolverNN (avg): {statistics.mean(n_errors_p_corrected_by_ns)}%)")   
    # print(f"Num errors of the perception corrected by the Mask-Predictor (avg): {statistics.mean(n_error_perception_masked)}%)")   
    print("\n---\n")
    print(f"Average number of hints: {np.mean(n_hints)}, Average number of solutions: {np.mean(n_solutions)}")
    print(f"Average number of solverNN errors out of 81 cells: {np.mean(n_error_ns)} errors, {np.mean(n_error_ns)/81}%") #added
    print(f"Average percentage of solverNN hint errors out of all hint: {np.mean(perc_error_ns_i)}%)") #added
    print(f"Average percentage of solverNN solution errors out of all solutions: {np.mean(perc_error_ns_s)}%)")#added
    print(f"Average percentage of solverNN hint errors masked out of all incorrect hints: {np.mean(perc_error_ns_masked_i)}%)")   
    print(f"Average percentage of solverNN solution errors masked out of all incorrect solutions: {np.mean(perc_error_ns_masked_s)}%)")
    print(f"Average percentage of SolverNN correct solutions being preserved by Mask-Predictor: {np.mean(perc_correct_ns_preserved_s)}%")
    print(f"Average percentage of SolverNN correct hints being preserved by Mask-Predictor: {np.mean(perc_correct_ns_preserved_i)}%")
    print(f"Average percentage of cells masked by Mask-Predictor(avg): {np.mean(perc_cells_masked)}%")
    # print(f"Average percentage of solverNN hint errors corrected out of all incorrect hint (avg): {np.mean(perc_error_ns_corrected_i)}%)")   
    # print(f"Average percentage of solverNN solution errors corrected out of all incorrect solutions(avg): {np.mean(perc_error_ns_corrected_s)}%)")
    performance_dict['avg_p_hint_errors'] = np.mean(perc_error_ns_i)
    performance_dict['avg_p_solution_errors'] = np.mean(perc_error_ns_s)
    performance_dict['avg_p_solution_errors_masked'] = np.mean(perc_error_ns_masked_s)
    performance_dict['avg_p_input_errors_masked'] = np.mean(perc_error_ns_masked_i)
    performance_dict['avg_p_correct_input_preserved'] = np.mean(perc_correct_ns_preserved_i)
    performance_dict['avg_p_correct_solution_preserved'] = np.mean(perc_correct_ns_preserved_s)
    performance_dict['avg_p_cells_masked'] = np.mean(perc_cells_masked)
    return performance_dict

def eval_improvement(y_pred,y_pred_b,y_target,dataset_name=None):
    time_solutions = []
    # y_pred = [l.tolist()for l in y_pred] 
    y_pred_b = [l.tolist()for l in y_pred_b]
    # y_pred = [list(map(int,i)) for i in y_pred] #Prolog solution
    y_pred_b = [list(map(int,i)) for i in y_pred_b] #SolverNN solution
    y_targ = []
    for board in y_target:
        tmp_board = []
        for cell in board:
            tmp_board.append(cell.argmax())
        y_targ.append(tmp_board)
    y_targ = [list(map(int,i)) for i in y_targ]
    num_total = len(y_pred_b)
    num_correct_b =0
    num_correct = 0
    num_corrected_boards = 0
    num_wrong_boards = 0
    average_correct_cells = []
    for i in range(len(y_pred_b)):
        correct_cells=0.0
        if y_pred_b[i] == y_targ[i]:
            num_correct_b += 1
        else:
            if dataset_name == 'multiple_sol':
                input_board = retrieve_hints_from_solution(np.array(y_targ[i]),dataset_name)
                check_input = check_input_board(input_board,np.array(y_pred_b[i]))
                consistent = check_consistency_board(np.array(y_pred_b[i]))
                if check_input and consistent:
                    num_correct_b += 1
                    #print('alternative solution found')
                else:
                  if y_pred[i] == y_targ[i]:
                    num_corrected_boards += 1  
            # else:
            #     if y_pred[i] == y_targ[i]:
            #         num_corrected_boards += 1
        
        # if y_pred[i] == y_targ[i]:
        #     num_correct += 1
        #     time_solutions.append(1)
        # else:
        #     time_solutions.append(0)
        #     if y_pred_b[i] == y_targ[i]:
        #         num_wrong_boards += 1
        # for j,k in zip(y_pred[i],y_targ[i]):
        #     if j==k:
        #         correct_cells+=1
        # correct_cells/=81
        # average_correct_cells.append(correct_cells)
    print(f"* Num. of correct solution boards from Neuro-Solver: {num_correct_b}/{num_total}-- ({(num_correct_b*100.)/(num_total):.2f}%)")
    # print(f"* Num. of correct solution boards from NASR: {num_correct}/{num_total}-- ({((num_correct)*100.)/(num_total):.2f}%)")
    # if (num_total-num_correct_b) >0 :
    #     print(f"* Num. of wrong solution boards (from Neuro-solver) \n  that have been corrected with NASR: {num_corrected_boards}/{num_total-num_correct_b} -- ({(num_corrected_boards*100.)/(num_total-num_correct_b):.2f}%)")
    # if num_correct_b>0 :
    #     print(f"* Num. of correct solution boards (from Neuro-solver) \n  that have been corrupted with NASR: {num_wrong_boards}/{num_correct_b} -- ({num_wrong_boards*100./num_correct_b:.2f}%)")
    # print(f"* Num correct cells (avg): {statistics.mean(average_correct_cells)}%")   
    return time_solutions

def analyze_confidence(test_loader, model, args, temperatures):
    # performance_dict['avg_solvernn_confidence'] 
    # performance_dict['avg_mask_confidence']
    # performance_dict['temp'] 
    # performance_dict['solvernn_hint_cf']              # performance_dict['mask_hint_cf'] 
    # performance_dict['solvernn_sol_cf']               # performance_dict['mask_sol_cf']
    # performance_dict['solvernn_correct_sol_cf']       # performance_dict['mask_correct_sol_cf'] 
    # performance_dict['solvernn_error_sol_cf']         # performance_dict['mask_error_sol_cf']
    # performance_dict['solvernn_catch_error_sol_cf']   # performance_dict['mask_catch_error_sol_cf'] 
    # performance_dict['solvernn_miss_error_sol_cf']    # performance_dict['mask_miss_error_sol_cf']
    # performance_dict['avg_p_hint_errors']               # performance_dict['avg_p_solution_errors'] 
    # performance_dict['avg_p_solution_errors_masked']    # performance_dict['avg_p_input_errors_masked']
    # performance_dict['avg_p_correct_input_preserved']   # performance_dict['avg_p_correct_solution_preserved'] 
    # performance_dict['avg_p_cells_masked']
    
    performance_dict_list = []
    print(f"Start Manipulate {args.performance_mask}")
    for temp in temperatures:
        args.temp = temp
        performance_dict_list.append(validate(test_loader, model, args))

    out_file = f'outputs/confidence/{args.data}/{args.data}_{args.performance_mask}_cf_results'
    
    np.save(out_file, performance_dict_list)
    print(f"Mask:{args.performance_mask}")
    print("Saved list of performance dictionaries to file")

    
def main():
    parser = init_parser()
    args = parser.parse_args()
    # ---------------------------------- checkpoint_best
    #ckpt_path = f'outputs/rl/{args.data}/checkpoint_best_L.pth' # usually worse
    ckpt_path = f'outputs/rl/{args.data}/checkpoint_best_R.pth' 
    # ----------------------------------
    # image_path = f'outputs/confidence/{args.data}'
    print(args.analysis)
    if os.path.isfile(args.noise_setting):
        with open(args.noise_setting) as f:
            noise_setting = json.load(f)
    else:
        noise_setting = {"noise_type": "awgn", "snr": -0.5}
    noise_setting = str(noise_setting).replace(' ', '').replace("'", "")
   
    if args.transform_data:
        test_dataset = SudokuDataset_Solver_Inter(args.data,'-test', nasr=args.nasr, transform=args.transform_data,t_param=args.transform_data_param)
    else:
        test_dataset = SudokuDataset_Solver_Inter(args.data,'-test', nasr=args.nasr)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    # Model

    model = get_no_perc_model(block_len=args.block_len)
    if args.nasr == 'rl':
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))  #to check preformance of the RL
    else:
        assert args.nasr == 'pretrained', f'{args.nasr} not supported, choose between pretrained and rl'
        model.load_pretrained_models(args.data, args.pos_weights) #to check preformance of the pretrained pipeline without RL
    model.to(args.gpu_id)
    # Main loop
    
    print(f"Begin evaluation on {args.data}")
    if args.analysis == 1:
        temperatures = [0.5, 1, 10/9.5, 10/9, 10/8.5, 10/8, 10/7, 2, 10/3, 10]
        
        analyze_confidence(test_loader, model, args, temperatures)
    else: 
        print(args.temp)
        performance_dict = validate(test_loader, model, args)
    print("Evaluation Complete!")
    
 

if __name__ == '__main__':

    main()
