#!/bin/bash
#SBATCH --job-name=finetune_reg_custom
#SBATCH --nodes=1            # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH -t 1-12:00
#SBATCH --mem=320GB
#SBATCH --array=180-269%8
#SBATCH -o out/custom_large/transfer_%A_%a.out
#SBATCH -e out/custom_large/transfer_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature
 
# train resnet
model='50'     # width of intermediate layer 
batchsize=64

# get seed and lam
seed=$((SLURM_ARRAY_TASK_ID % 5 + 400))

# select alpha for l2sp
alpha=0.1

# select regularization strength
if [ $(((SLURM_ARRAY_TASK_ID / 5) % 2)) == 0 ]; then
    lam=0.01
else
    lam=0.001
fi

# select regularization type
if [ $(((SLURM_ARRAY_TASK_ID / 10) % 3)) == 0 ]; then
    # reg="l2sp bss"
    reg="bss"
    train_modifier="--reg-freq-update 1"
    burnin=0
elif [ $(((SLURM_ARRAY_TASK_ID / 10) % 3)) == 1 ]; then
    # reg="l2sp spectral"
    reg="spectral"
    train_modifier="--reg-freq-update 160 --iterative"
    burnin=30
else
    # reg="l2sp eig-ub"
    reg="eig-ub"
    train_modifier="--reg-freq-update 160 --iterative"
    burnin=30
fi

if [ $(((SLURM_ARRAY_TASK_ID / 30) % 3)) == 0 ]; then
    # base
    tag='transfer_cifar10_resize_custom'
    model_path=""/n/pehlevan_lab/Users/shengy/geometry/GeomNet/model/pretrain/imagenet1k_m50_bs128_lr0.02_optSGD_wd0.0001_mom0.9_nlReLU_regNone_e120_seed0/model_e120.pt""
elif [ $(((SLURM_ARRAY_TASK_ID / 30) % 3)) == 1 ]; then
    # reg 0.01
    tag='transfer_cifar10_resize_custom_reg0.01'
    model_path="/n/pehlevan_lab/Users/shengy/geometry/GeomNet/model/pretrain/imagenet1k_m50_bs128_lr0.02_optSGD_wd0.0001_mom0.9_nlReLU_lam0.01_reg['eig-ub']_e120_b80_seed0_rf1_ru80/model_e120.pt"
else
    # reg 0.001
    tag='transfer_cifar10_resize_custom_reg0.001'
    model_path="/n/pehlevan_lab/Users/shengy/geometry/GeomNet/model/pretrain/imagenet1k_m50_bs128_lr0.02_optSGD_wd0.0001_mom0.9_nlReLU_lam0.001_reg['eig-ub']_e120_b80_seed0_rf1_ru80/model_e120.pt"
fi

# datasets & optimal learning rates
if [ $(((SLURM_ARRAY_TASK_ID / 5) % 3)) == 0 ]; then
    # * dog
    epochs=50       # epochs 
    lr="0.005"      # learning rate 
    data="dog"
    log_epoch=10
    log_model=25
elif [ $(((SLURM_ARRAY_TASK_ID / 5) % 3)) == 1 ]; then
    # * flower
    epochs=1000     # epochs 
    lr="0.005"      # learning rate 
    data="flower"
    log_epoch=100
    log_model=250
else
    # * indoor
    epochs=100
    lr='0.01'
    data="indoor"
    log_epoch=10
    log_model=50
fi




# # datasets & optimal learning rates
# data='cifar10_resized'
# epochs=50
# log_epoch=1
# log_model=10

# learning rate
# lr="0.01"      # learning rate 
wd="1e-4"      # ! weight decay !

# train_modifier="--max-layer 2"
# train_modifier="--reg-freq-update 15"
# train_modifier="--reg-freq-update 24 --max-layer 2"
# train_modifier=""

python src/train_reg_finetune.py --seed $seed --model $model --epochs $epochs --lr $lr \
 --data $data --batch-size $batchsize --log-epoch $log_epoch --log-model $log_model --burnin $burnin \
 --lam $lam --alpha $alpha --reg $reg --tag $tag $train_modifier --schedule --custom-pretrain --pretrain-wt-path $model_path

python src/eval_black_box_robustness_finetune.py \
--model $model --epochs $epochs --alpha $alpha \
--seed $seed --lr $lr --wd $wd --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg $reg --burnin $burnin \
--vmin -3 --vmax 3 --perturb-vmin -0.5 --perturb-vmax 0.5 \
--eval-epoch $epochs --eval-sample-size 500 --reg-freq 1 \
$train_modifier --adv-batch-size 4 --custom-pretrain
