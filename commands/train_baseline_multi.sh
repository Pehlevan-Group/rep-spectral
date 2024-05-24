#!/bin/bash
#SBATCH -c 2               
#SBATCH -t 1-00:00           
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH -o out/mnist_base_%j.out   
#SBATCH -e out/mnist_base_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: baseline
# setup parameters
tag='mnist'
w=2000

for seed in "401" "402" "403" "404" "405" "501"; do
    for data in "mnist" "fashion_mnist"; do
        for wd in "0" "1e-4"; do
            
            # long training
            python src/train_reg_multi.py \
                --data $data \
                --hidden-dim $w \
                --reg None \
                --lr 0.001 \
                --wd $wd 
                --epochs 200 \
                --log-epoch 10 \
                --log-model 10 \
                --tag $tag \
                --seed $seed
        done
    done
done
