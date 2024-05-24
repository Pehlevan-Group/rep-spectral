#!/bin/bash
#SBATCH --job-name=simclr_base
#SBATCH --nodes=1            # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH -t 2-00:00
#SBATCH --mem=200GB
#SBATCH --array=0-4%5
#SBATCH -o out/simclr/simclr_%A_%a.out
#SBATCH -e out/simclr/simclr_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train simclr
model='18'      # width of intermediate layer 
epochs=1000     # epochs 
# lr="0.01"       # learning rate 
nl="ReLU"

# seed=$1
# weight_decay=$2  # ! weight decay 

data="cifar10"
batchsize="512"

log_epoch=50
log_model=50
tag='simclr18'

# reg setup 
reg="None"
# lam=$4
burnin=900

# train modifier
train_modifier="--reg-freq-update 1"

# get seed and lam
lam_idx=$((SLURM_ARRAY_TASK_ID % 5))
seed=$((lam_idx + 400))


# set parameters
# if [ $(($SLURM_ARRAY_TASK_ID / 5)) == 0 ]; then
#     lr=5e-3
#     weight_decay=5e-4
# elif [ $(($SLURM_ARRAY_TASK_ID / 5)) == 1 ]; then
#     lr=1e-2
#     weight_decay=5e-4
# elif [ $(($SLURM_ARRAY_TASK_ID / 5)) == 2 ]; then
#     lr=1e-2
#     weight_decay=1e-4
# fi

# add missing parts
lr=1e-2
weight_decay=5e-4

# select regularization strength
lam=0.01

# # train 
# python src/train_reg_simclr.py --model $model --epochs $epochs \
# --seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
# --batch-size $batchsize --tag $tag \
# --log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
# $train_modifier

# evaluation
python src/eval_black_box_robustness_contrastive.py --model-type simclr \
--model $model --epochs $epochs \
--seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
--vmin 0 --vmax 1 --perturb-vmin -0.3 --perturb-vmax 0.3 \
--eval-epoch $epochs --eval-sample-size 1000 --reg-freq 1 \
$train_modifier
