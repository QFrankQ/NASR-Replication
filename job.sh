#!/bin/bash

# Request a GPU partition node

#SBATCH -G 1                  # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:1          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=1   # This needs to match num GPUs. default 8
#SBATCH --mem=20000          # Requested Memory
#SBATCH -p gpu-preempt     # Partition
#SBATCH -t 08:00:00              # Job time limit
#SBATCH -o slurm-%j.out       # %j = job ID


# Change to the directory where the code is located

# Activate the conda environment
#module load miniconda/22.11.1-1
eval "$(conda shell.bash hook)"
conda activate nasr

echo "Environment is ready!"

# Run the python script
# python src/test.py
# python src/datasets_inter.py
# python src/eval_confidence.py  --nasr rl --solver prolog --data minimal_17 --temp 0.8 --gpu-id 0 --performance-mask unmask_error_sol_cells_indices
# python src/eval_confidence.py  --nasr rl --solver prolog --data big_kaggle --analysis 1 --gpu-id 0 --performance-mask error_sol_cells_indices
# python src/plot_confidence_performance.py --data minimal_17 
# python src/plot_confidence_performance.py --data minimal_17 --performance-mask correct_sol_cells_indices
# python src/plot_confidence_performance.py --data minimal_17 --performance-mask unmask_error_sol_cells_indices
python src/plot_confidence_performance.py --data minimal_17 --performance-mask error_sol_cells_indices
# python src/plot_confidence_performance.py --data big_kaggle --performance-mask correct_sol_cells_indices
# python src/plot_confidence_performance.py --data big_kaggle --performance-mask unmask_error_sol_cells_indices
python src/plot_confidence_performance.py --data big_kaggle --performance-mask error_sol_cells_indices
# python src/plot_confidence_performance.py --data big_kaggle
# python src/eval_transformer_models.py --module mask --data minimal_17 --gpu-id 0
# python src/train_transformer_models.py --module mask --data minimal_17 --pos-weights 100 --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0
# python src/eval_pipeline_no_prolog.py --nasr pretrained --solver prolog --data minimal_17 --gpu-id 0 
# python src/train_transformer_models-c.py --module solvernn --data minimal_17 --epochs 50 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0 --file-name klLoss --constraint-loss kl --reg-scale 0.00001
# python src/train_transformer_models-c.py --module solvernn --data big_kaggle --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0 --file-name cosineLoss --constraint-loss cosine --reg-scale 0.0003
# python src/train_transformer_models-c.py --module solvernn --data minimal_17 --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0 --file-name cosineLoss --constraint-loss cosine --reg-scale 0.00005
# python src/train_transformer_models-c.py --module solvernn --data minimal_17 --epochs 200 --warmup 10 --batch-size 128 --lr 0.0001 --weight-decay 3e-1 --clip-grad-norm 1 --gpu-id 0 --file-name standard
echo "Job is done!"
