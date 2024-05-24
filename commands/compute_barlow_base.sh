#!/bin/bash
#SBATCH --job-name=barlow_compute_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 0-06:00
#SBATCH --mem=80GB
#SBATCH --array=0,1,5,6%2
#SBATCH -o out/barlow_relu_compute_base_%A_%a.out
#SBATCH -e out/barlow_relu_compute_base_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train barlow
model='18'      # width of intermediate layer 
epochs=1000     # epochs 
lr="0.01"       # learning rate 
nl="ReLU"

lam_idx=$((SLURM_ARRAY_TASK_ID % 5))
seed=$((lam_idx + 400))

if [ $((SLURM_ARRAY_TASK_ID / 5)) == 0 ]; then 
    weight_decay="0"
else 
    weight_decay="1e-4"
fi 

data="cifar10"
batchsize="1024"
projector="256"

log_epoch=50
log_model=50
tag='barlow_relu'

# reg setup 
reg="None"
lam=0.1
burnin=900

# train modifier
# train_modifier="--reg-freq-update 10"
train_modifier=""

# compute 
sample_step=40
scan_batchsize=4

python src/compute_reg_barlow.py --model $model --epochs $epochs --projector $projector \
--seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
$train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
