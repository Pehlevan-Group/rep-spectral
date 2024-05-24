#!/bin/bash
#SBATCH -c 2
#SBATCH -t 1-08:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -o out/barlow/barlow_gelu_%j.out
#SBATCH -e out/barlow/barlow_gelu_%j.err
#SBATCH --mail-type=END,FAIL

# train barlow
model='34'      # width of intermediate layer 
epochs=1000     # epochs 
lr="0.01"       # learning rate 
nl="ReLU"
weight_decay="1e-4"  # ! weight decay 
data="cifar10"
batchsize="1024"
projector="256"

log_epoch=50
log_model=50
tag='barlow_relu'

# reg setup 
lam="0.01"
reg="None"
burnin=900

# for seed in "400" "500" "600" "700" "800"; do
for seed in "400"; do
# for seed in "40" "50" "60" "70" "80"; do
    # base model
    python src/train_reg_barlow.py --model $model --epochs $epochs --projector $projector \
    --seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
    --batch-size $batchsize --tag $tag \
    --log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin 
done
