#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-16:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -o out/resnet/resnet_relu_%j.out
#SBATCH -e out/resnet/resnet_relu_%j.err
#SBATCH --mail-type=END,FAIL

# train resnet
seeds=$1
model='18'      # width of intermediate layer 
epochs="200"    # epochs 
lr="0.01"       # learning rate 
nl="ReLU"
weight_decay=$2  # ! weight decay 
data="cifar10"
batchsize="1024"

log_epoch=5
log_model=20
tag='resnet_relu18'

# reg setup 
lam=$4
reg=$3
burnin=160

# train_modifier=""
# train_modifier="--max-layer 2"
# train_modifier="--reg-freq-update 24"
# train_modifier="--reg-freq-update 24 --max-layer 2"
train_modifier=$5


# train regularization 
for seed in $seeds; do
    python src/train_reg_resnet.py --model $model --epochs $epochs \
    --seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
    --batch-size $batchsize --tag $tag \
    --log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin  \
    $train_modifier # ! modify regularization 
done

# evaluate
for seed in $seeds; do
    python src/eval_black_box_robustness.py --model $model --epochs $epochs \
    --seed $seed --lr $lr --nl $nl --wd $weight_decay --data $data \
    --batch-size $batchsize --tag $tag \
    --log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
    --vmin -3 --vmax 3 --perturb-vmin -0.3 --perturb-vmax 0.3 \
    --eval-epoch $epochs --eval-sample-size 1000 --reg-freq 1 \
    $train_modifier # ! modify regularization 
done
