#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-12:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o out/sin_gelu_%j.out
#SBATCH -e out/sin_gelu_%j.err

# sin-random training with SLP
epochs=15000
burnin=10500
log_model=1500
log_epoch=750

lr=0.05
wd=1e-5
bs=64
nl="GELU"
lam=1e-3

tag="sin_deep_gelu"

# start training
for seed in "400" "500" "600" "700" "800"; do
    for hd in "8 8 8" "20 20 20"; do
        # # base model
        # python src/train_reg.py \
        # --data sin-random --step 40 --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed
        
        # # reg model
        # python src/train_reg.py \
        # --data sin-random --step 40 --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed --iterative
        
        # # spectral model
        # python src/train_reg.py \
        # --data sin-random --step 40 --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed --iterative
        
        # # reg model
        # python src/train_reg.py \
        # --data sin-random --step 40 --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed
        
        # # spectral model
        # python src/train_reg.py \
        # --data sin-random --step 40 --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed

        # change max layer 1
        python src/train_reg.py \
        --data sin-random --step 40 --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative --max-layer 1

        python src/train_reg.py \
        --data sin-random --step 40 --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --max-layer 1

        # change max layer 2
        python src/train_reg.py \
        --data sin-random --step 40 --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative --max-layer 2

        python src/train_reg.py \
        --data sin-random --step 40 --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --max-layer 2
    done
done
