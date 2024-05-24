#!/bin/bash
#SBATCH -c 2
#SBATCH -t 1-08:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -o out/barlow_gelu_%j.out
#SBATCH -e out/barlow_gelu_%j.err
#SBATCH --mail-type=END,FAIL

# train barlow
model='18'      # width of intermediate layer 
epochs=1000     # epochs 
lr="0.01"       # learning rate 
nl="ReLU"

seed=$1
weight_decay=$2  # ! weight decay 

data="cifar10"
batchsize="1024"
projector="256"

log_epoch=50
log_model=50
tag='barlow_relu'

# reg setup 
reg=$3
lam=$4
burnin=900

# train modifier
# train_modifier="--reg-freq-update 10"
train_modifier=$5

# train 
python src/train_reg_barlow.py --model $model --epochs $epochs --projector $projector \
--seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
$train_modifier

# evaluation
python src/eval_black_box_robustness_contrastive.py --model-type barlow \
--model $model --epochs $epochs --projector $projector \
--seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
--vmin -3 --vmax 3 --perturb-vmin -0.3 --perturb-vmax 0.3 \
--eval-epoch $epochs --eval-sample-size 1000 --reg-freq 1 \
$train_modifier
