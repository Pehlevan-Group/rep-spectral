#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-8:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH -o out/mnist_gelu_eval_%j.out
#SBATCH -e out/mnist_gelu_eval_%j.err

# sin-random training with SLP
epochs=200
burnin=180
log_model=20
log_epoch=10

lr=0.1
wd=0 # ! tune weight decay
bs=1024
nl="GELU"
lam=0.001

tag="mnist_gelu"
data="mnist"

# start evaluation
for seed in "400" "500" "600" "700" "800"; do
    for hd in "2000"; do
        # # base model
        # python src/eval_black_box_robustness.py \
        # --data $data --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed \
        # --eval-epoch $epochs --eval-sample-size 3000 --reg-freq 1

        # # reg model
        # python src/eval_black_box_robustness.py \
        # --data $data --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin --reg-freq 1 \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed --iterative \
        # --eval-epoch $epochs --eval-sample-size 3000 
        
        # spectral model
        python src/eval_black_box_robustness.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed --iterative \
        --eval-epoch $epochs --eval-sample-size 3000 
        
        # # reg model
        # python src/eval_black_box_robustness.py \
        # --data $data --batch-size $bs \
        # --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin $burnin --reg-freq 1 \
        # --epochs $epochs --tag $tag \
        # --log-epoch $log_epoch --log-model $log_model \
        # --seed $seed \
        # --eval-epoch $epochs --eval-sample-size 3000 
        
        # spectral model
        python src/eval_black_box_robustness.py \
        --data $data --batch-size $bs \
        --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin $burnin --reg-freq 1 \
        --epochs $epochs --tag $tag \
        --log-epoch $log_epoch --log-model $log_model \
        --seed $seed \
        --eval-epoch $epochs --eval-sample-size 3000 
    done
done
