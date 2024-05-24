#!/bin/bash
#SBATCH --job-name=transfer_base
#SBATCH --nodes=1            # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH -t 0-12:00
#SBATCH --mem=600GB
#SBATCH --array=0-119%15
#SBATCH -o out/transfer/transfer_%A_%a.out
#SBATCH -e out/transfer/transfer_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train resnet
model='34'      # width of intermediate layer 
batchsize=128
epochs=200
data="cifar100"
log_epoch=1
log_model=30

# get seed and lam
seed=$((SLURM_ARRAY_TASK_ID % 10 + 400))

# testing learning rates
lr="0.1"
wd=5e-4

# specify paths
tag='transfer'

# lam setup 
if [ $(((SLURM_ARRAY_TASK_ID / 10) % 3)) == 0 ]; then
    lam=0.01
elif [ $(((SLURM_ARRAY_TASK_ID / 10) % 3)) == 1 ]; then
    lam=0.001
else
    lam=0.0001
fi 

# setup burnin
if [ $(((SLURM_ARRAY_TASK_ID / 30) % 2)) == 0 ]; then
    burnin=160
else 
    burnin=120
fi

# setup training modifier
if [ $(((SLURM_ARRAY_TASK_ID / 60) % 2)) == 0 ]; then
    train_modifier="--reg-freq-update 5"
else 
    train_modifier="--reg-freq-update 20"
fi

# train_modifier="--max-layer 2"
# train_modifier="--reg-freq-update 15"
# train_modifier="--reg-freq-update 24 --max-layer 2"
# train_modifier=""

python src/train_reg_transfer.py --seed $seed --model $model --epochs $epochs --lr $lr --wd $wd \
 --data $data --batch-size $batchsize --log-epoch $log_epoch --log-model $log_model --burnin $burnin \
 --lam $lam --reg eig-ub --tag $tag $train_modifier

# TODO: setup evaluation
# adv_batch_size=2
# python src/eval_black_box_robustness_transfer.py \
# --model $model --epochs $epochs \
# --seed $seed --lr $lr --data $data \
# --batch-size $batchsize --tag $tag \
# --log-model $log_model --log-epoch $log_epoch --lam $lam --reg None --burnin $burnin \
# --vmin -3 --vmax 3 --perturb-vmin -0.3 --perturb-vmax 0.3 \
# --eval-epoch $epochs --eval-sample-size 500 --reg-freq 1 \
# $train_modifier --adv-batch-size $adv_batch_size
