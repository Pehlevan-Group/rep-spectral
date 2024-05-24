#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-24:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o out/fmnist_gelu_%j.out
#SBATCH -e out/fmnist_gelu_%j.err

# sin-random training with SLP
epochs=400
burnin=360
log_model=40
log_epoch=10

lr=0.1
wd=1e-4
bs=1024
nl="GELU"
lam=1e-3

tag="fashion_mnist_long_gelu"
data="fashion_mnist"

# start training
for seed in "400" "500" "600" "700" "800"; do
    for hd in "2000"; do
        # base model
        python src/train_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed
        
        # reg model
        python src/train_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative
        
        # spectral model
        python src/train_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative
        
        # reg model
        python src/train_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed
        
        # spectral model
        python src/train_reg_multi.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed
    done
done
