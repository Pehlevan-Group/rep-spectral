#!/bin/bash
#SBATCH --job-name=barlow_compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 1-00:00
#SBATCH --mem=80GB
#SBATCH --array=2,4,5,7,44,45,46%2
#SBATCH -o out/barlow_gelu_%A_%a.out
#SBATCH -e out/barlow_gelu_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train barlow
model='18'      # width of intermediate layer 
epochs=1000     # epochs 
lr="0.01"       # learning rate 
nl="ReLU"

# seed=$1
# weight_decay=$2  # ! weight decay 

data="cifar10"
batchsize="1024"
projector="256"

log_epoch=50
log_model=50
tag='barlow_relu'

# reg setup 
reg="eig-ub"
# lam=$4
burnin=900

# train modifier
# train_modifier="--reg-freq-update 10"
# train_modifier=$5

# get seed and lam
lam_idx=$((SLURM_ARRAY_TASK_ID % 40))
seed=$((lam_idx / 4 + 400))

# set parameters
if [ $(($lam_idx % 4)) == 0 ]; then
    weight_decay=0.
    train_modifier="--reg-freq-update 24"
elif [ $(($lam_idx % 4)) == 1 ]; then
    weight_decay=0.
    train_modifier="--reg-freq-update 10"
elif [ $(($lam_idx % 4)) == 2 ]; then
    weight_decay=1e-4
    train_modifier="--reg-freq-update 24"
else
    weight_decay=1e-4
    train_modifier="--reg-freq-update 10"
fi

# select regularization strength
if [ $((SLURM_ARRAY_TASK_ID / 40)) == 0 ]; then
    lam=0.01
else
    lam=0.001
fi

sample_step=40
scan_batchsize=2
# compute quantities 
python src/compute_reg_barlow.py --model $model --epochs $epochs --projector $projector \
--seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
$train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
