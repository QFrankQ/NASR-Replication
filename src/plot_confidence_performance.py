import numpy as np
import matplotlib.pyplot as plt
import argparse

def init_parser():
    parser = argparse.ArgumentParser(description='Quick testing script')

    # General args
    parser.add_argument('--nasr', type=str, default='rl',
                        help='choice of nasr with nasr_rl or pretrained (without rl)')
    parser.add_argument('--data', type=str, default='big_kaggle',
                        help='dataset name between big_kaggle, minimal_17, multiple_sol and satnet')
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

def plot_confidence_performance(
    performance_dict_list,
    outfile,
    performance_categories_1,
    performance_categories_2,
    x_label,
    title,
    metric1_label,
    metric2_label,
    metric1_colors,
    metric2_colors,
    metric1_ylim = None,
    metric2_ylim = None,
    args=None
):
    fig, ax1 = plt.subplots(figsize=(7, 6))
    temps = [d['temp'] for d in performance_dict_list]

    # Plot curves for Metric 1 (on primary y-axis)
    for i, cat in enumerate(performance_categories_1):
        metric_curve = [d[cat] for d in performance_dict_list]
        print(f"metric: {cat}, curve: {metric_curve}")
        ax1.plot(temps, metric_curve, metric1_colors[i], label=cat)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(metric1_label, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Place the first legend in the lower left
    ax1.legend(loc='lower left')  # Move outside the plot
    if metric1_ylim:
        ax1.set_ylim(metric1_ylim)

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Plot curves for Metric 2 (on secondary y-axis)
    for i, cat in enumerate(performance_categories_2):
        metric_curve = [d[cat] for d in performance_dict_list]
        print(f"performance mask: {args.performance_mask}, metric: {cat}, curve: {metric_curve}")
        ax2.plot(temps, metric_curve, metric2_colors[i], label=cat)
    ax2.set_ylabel(metric2_label, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Place the second legend in the lower right, outside the plot
    ax2.legend(loc='lower left', bbox_to_anchor=(0, 0.2))  # Adjusted position
    if metric2_ylim:
        ax2.set_ylim(metric2_ylim)

    # Add title
    plt.title(title)
    plt.xscale('log')

    # Adjust layout to ensure space for legends
    # plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave extra space at the bottom for legends
    plt.tight_layout()
    # Save the plot
    plt.savefig(outfile)
    

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
def plot_solution_confidence(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_correct_sol_cf', 'solvernn_catch_error_sol_cf', 'solvernn_miss_error_sol_cf']
    mask_cf_list = ['mask_correct_sol_cf', 'mask_catch_error_sol_cf', 'mask_miss_error_sol_cf']
    title = f'SolverNN & Mask Solution Confidence(CF)-{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask CF, Maximum val: log(2)'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-np.log(2)-0.1, 0.05*np.log(2))
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_cf_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("Solution Confidence Plotted")

def plot_general_confidence(performance_dict_list, args):
    solvernn_cf_list = ['avg_solvernn_confidence', 'solvernn_hint_cf', 'solvernn_sol_cf']
    mask_cf_list = ['avg_mask_confidence', 'mask_hint_cf', 'mask_sol_cf']
    title = f'SolverNN & Mask General Confidence(CF)-{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask CF, Maximum val: log(2)'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-np.log(2)-0.1, 0.05*np.log(2))
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_cf_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("General Confidence Plotted")

def plot_unmasked_err_sol_cf_vs_acc(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_miss_error_sol_cf']
    # mask_acc_list = ['avg_p_solution_errors_masked','avg_p_correct_input_preserved','avg_p_correct_solution_preserved']
    mask_acc_list = ['avg_p_solution_errors','avg_p_solution_errors_masked','avg_p_correct_solution_preserved','avg_p_correct_input_preserved']
    title = f'SolverNN Confidence(CF) in Missed Error Solution Cells vs Mask Performance -{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask Performance, x100%'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-0.05, 1.05)
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_acc_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("SolverNN CF in missed error sol Cell vs % Error Sol masked Plotted")
    
def plot_correct_sol_cf_vs_acc(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_correct_sol_cf']
    mask_acc_list = ['avg_p_solution_errors', 'avg_p_solution_errors_masked', 'avg_p_correct_solution_preserved', 'avg_p_correct_input_preserved']
    title = f'SolverNN Confidence(CF) in Correct Solution Cells vs Mask Performance -{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask Performance, x100%'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-0.05, 1.05)
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_acc_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("SolverNN CF in Correct sol Cell vs % Error Sol masked Plotted")
    
def plot_error_sol_cf_vs_acc(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_error_sol_cf']
    mask_acc_list = ['avg_p_solution_errors', 'avg_p_solution_errors_masked', 'avg_p_correct_solution_preserved', 'avg_p_correct_input_preserved']
    title = f'SolverNN Confidence(CF) in Error Solution Cells vs Mask Performance -{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask Performance, x100%'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-0.05, 1.05)
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_acc_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("SolverNN CF in Correct sol Cell vs % Error Sol masked Plotted")
    
def plot_mod_correct_solution_confidence(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_correct_sol_cf', 'solvernn_catch_error_sol_cf', 'solvernn_miss_error_sol_cf']
    mask_cf_list = ['mask_correct_sol_cf', 'mask_catch_error_sol_cf', 'mask_miss_error_sol_cf']
    title = f'SolverNN & Mask Mod Correct Solution Confidence(CF)-{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask CF, Maximum val: log(2)'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-np.log(2)-0.1, 0.05*np.log(2))
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_cf_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("Mod Correct Solution Confidence Plotted")

def plot_mod_error_solution_confidence(performance_dict_list, args):
    solvernn_cf_list = ['solvernn_error_sol_cf']
    mask_cf_list = ['mask_correct_sol_cf', 'mask_catch_error_sol_cf', 'mask_miss_error_sol_cf']
    title = f'SolverNN & Mask Mod Error Solution Confidence(CF)-{args.data}'
    xlabel = 'Temperature'
    metric1_label = 'SolverNN CF, Maximum val: log(9)'
    metric2_label = 'Mask CF, Maximum val: log(2)'
    metric1_colors = ['b-', 'b--', 'b-.', 'b:']
    metric2_colors = ['r-', 'r--', 'r-.', 'r:']
    metric1_ylim = (-np.log(9)-0.1, 0.05*np.log(9))
    metric2_ylim = (-np.log(2)-0.1, 0.05*np.log(2))
    outfile = f'outputs/confidence/{args.data}/{title}'
    plot_confidence_performance(
        performance_dict_list,
        outfile,
        solvernn_cf_list,
        mask_cf_list,
        xlabel,
        title,
        metric1_label,
        metric2_label,
        metric1_colors,
        metric2_colors,
        metric1_ylim,
        metric2_ylim,
        args
    )
    print("Mod Error Solution Confidence Plotted")
    
def main():
    parser = init_parser()
    args = parser.parse_args()
    
    file_path = f"outputs/confidence/{args.data}/{args.data}_{args.performance_mask}_cf_results.npy"
    print(file_path)
    performance_dict_list = np.load(file_path, allow_pickle=True)
    # plot_general_confidence(performance_dict_list, args)
    # plot_solution_confidence(performance_dict_list, args)
    if args.performance_mask == 'correct_sol_cells_indices':
        plot_correct_sol_cf_vs_acc(performance_dict_list, args)
        plot_mod_correct_solution_confidence(performance_dict_list,args)
    if args.performance_mask == 'unmask_error_sol_cells_indices':
        plot_unmasked_err_sol_cf_vs_acc(performance_dict_list, args)
    if args.performance_mask == 'error_sol_cells_indices':
        plot_error_sol_cf_vs_acc(performance_dict_list, args)
        plot_mod_error_solution_confidence(performance_dict_list, args)
if __name__ == '__main__':

    main()
