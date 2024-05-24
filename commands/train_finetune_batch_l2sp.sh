#!/bin/bash
#SBATCH --job-name=finetune_l2sp_wd
#SBATCH --nodes=1            # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH -t 0-12:00
#SBATCH --mem=160GB
#SBATCH --array=0-19%4
#SBATCH -o out/transfer_cifar10_resize/transfer_%A_%a.out
#SBATCH -e out/transfer_cifar10_resize/transfer_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train resnet
model='50'      # width of intermediate layer 
batchsize=64

# get seed and lam
seed=$((SLURM_ARRAY_TASK_ID % 10 + 400))

# select alpha for l2sp
if [ $(((SLURM_ARRAY_TASK_ID / 10) % 2)) == 0 ]; then
    alpha=0.1
else
    alpha=0.01
fi

# # datasets & optimal learning rates
# if [ $((SLURM_ARRAY_TASK_ID / 10)) == 0 ]; then
#     # * dog
#     epochs=500      # epochs 
#     lr="0.005"      # learning rate 
#     data="dog"
#     log_epoch=10
#     log_model=250
# elif [ $((SLURM_ARRAY_TASK_ID / 10)) == 1 ]; then
#     # * flower
#     epochs=1000     # epochs 
#     lr="0.005"      # learning rate 
#     data="flower"
#     log_epoch=10
#     log_model=250
# else
#     # * indoor
#     epochs=500
#     lr='0.01'
#     data="indoor"
#     log_epoch=10
#     log_model=250
# fi

# datasets & optimal learning rates
data='cifar10_resized'
epochs=50
log_epoch=1
log_model=10

# learning rate
lr="0.01"      # learning rate 
wd="1e-4"      # ! weight decay

# specify paths
tag='transfer_cifar10_resize_wd'

# reg setup 
lam=0.01
burnin=0

train_modifier="--reg-freq-update 1"
# train_modifier="--max-layer 2"
# train_modifier="--reg-freq-update 15"
# train_modifier="--reg-freq-update 24 --max-layer 2"
# train_modifier=""

python src/train_reg_finetune.py --seed $seed --model $model --epochs $epochs --lr $lr --wd $wd \
 --data $data --batch-size $batchsize --log-epoch $log_epoch --log-model $log_model --burnin $burnin \
 --lam $lam --alpha $alpha --reg l2sp --tag $tag $train_modifier --schedule

python src/eval_black_box_robustness_finetune.py \
--model $model --epochs $epochs --alpha $alpha \
--seed $seed --lr $lr --wd $wd --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg l2sp --burnin $burnin \
--vmin -3 --vmax 3 --perturb-vmin -0.5 --perturb-vmax 0.5 \
--eval-epoch $epochs --eval-sample-size 500 --reg-freq 1 \
$train_modifier --adv-batch-size 3
