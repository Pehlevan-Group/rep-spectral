#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-6:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o out/mnist/mnist_%j.out
#SBATCH -e out/mnist/mnist_%j.err
#SBATCH --mail-type=END,FAIL

# MNIST training
seeds=$1
epochs=200
burnin=160
log_model=20
log_epoch=10

lr=0.1
wd=$2 # ! tune weight decay
bs=1024
nl="GELU"
lam=$3

tag="mnist_small_init"
data="mnist"

train_modifier=""
# train_modifier="--reg-freq-update 10"
# train_modifier="--reg-freq-update 24"

# element computations
sample_step=40
scan_batchsize=256

# start training
for seed in $seeds; do
# for seed in "40" "50" "60" "70" "80"; do
    for hd in "2000"; do
        # base model
        python src/compute_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed $train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # reg model
        python src/compute_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub-pure --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative $train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # spectral model
        python src/compute_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative $train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # # reg model
        # python src/compute_reg_multi.py \
        # --data $data --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub-pure --lam $lam --burnin $burnin --reg-freq 1 \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed $train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # # spectral model
        # python src/compute_reg_multi.py \
        # --data $data --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed $train_modifier --sample-step $sample_step --scan-batchsize $scan_batchsize
    done
done
