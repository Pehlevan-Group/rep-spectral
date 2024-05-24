#!/bin/bash
#SBATCH -c 2            
#SBATCH -t 0-08:00           
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o out/train_reg_multi_%j.out   
#SBATCH -e out/train_reg_multi_%j.err 
#SBATCH --mail-type=END,FAIL

data="mnist"
tag="mnist_approx"
wd=0
reg="spectral"

for seed in "401" "402" "403" "404" "405" "501"; do
# for seed in "401" "402" "403" ""; do
# for seed in "404" "405" "501"; do
# for seed in "402" "403" "404" "405" "501"; do
    # for ss in "0.6" "0.2"; do
    for ss in "0.2"; do
        # for burnin in "180" "160" "140" "120"; do 
        for burnin in "0"; do
            # for lam in "1e-5" "1e-4" "4e-3" "1e-3"; do
            # for lam in "1e-5" "1e-3"; do
            python src/train_reg_multi.py \
                --data $data \
                --lam 0.0001 \
                --hidden-dim 2000 \
                --reg $reg \
                --lr 0.001 \
                --sample-size $ss \
                --wd $wd \
                --epochs 200 \
                --burnin $burnin \
                --log-epoch 1 \
                --tag $tag \
                --seed $seed \
                --reg-freq 1 \
                --iterative # ! approx
            # done 

            python src/train_reg_multi.py \
                --data $data \
                --lam 0.001 \
                --hidden-dim 2000 \
                --reg $reg \
                --lr 0.001 \
                --sample-size $ss \
                --wd $wd \
                --epochs 200 \
                --burnin $burnin \
                --log-epoch 1 \
                --tag $tag \
                --seed $seed \
                --reg-freq 1 \
                --iterative # ! approx 
        done
    done 
done 
