#!/bin/bash
#SBATCH --job-name=resnet50_pt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH -t 3-00:00
#SBATCH --mem=200GB
#SBATCH -o out/pretrain/pt_%A.out
#SBATCH -e out/pretrain/pt_%A.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# torchrun --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE src/train_reg_pretrain.py --schedule --tag pretrain --wd 1e-4 --seed 0 --batch-size 128 --checkpoint 90

# eig-ub regularization (lam = 0.01)
torchrun --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE src/train_reg_pretrain.py --schedule --tag pretrain --wd 1e-4 --seed 0 --batch-size 128 --reg eig-ub --burnin 80 --reg-freq-update 80 --lam 1e-2 --iterative

# # eig-ub regularization (lam = 0.001)
# torchrun --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE src/train_reg_pretrain.py --schedule --tag pretrain --wd 1e-4 --seed 0 --batch-size 128 --reg eig-ub --burnin 80 --reg-freq-update 80 --lam 1e-3 --iterative


# # cut mix strategy for larger batch size
# torchrun --nnodes=1 --nproc_per_node=$SLURM_GPUS_ON_NODE src/train_reg_pretrain.py --schedule --tag pretrain --lr 0.1 --epochs 300 --log-epoch 75 --log-model 30 --wd 1e-4 --seed 0 --batch-size 256 --cutmix
