#!/bin/bash
#SBATCH -c 2            
#SBATCH -t 0-08:00           
#SBATCH --mem=20000
#SBATCH -o out/baseline_eval_bb_%j.out   
#SBATCH -e out/baseline_eval_bb_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: baseline
# setup parameters
tag='linear_deep_approx'
w=2000

# single-hidden layer
# for seed in "401" "402" "403" "404" "405" "501"; do
#     for data in "mnist" "fashion_mnist"; do
#         for wd in "0" "1e-4"; do
#             python src/eval_black_box_robustness.py \
#                 --data $data \
#                 --hidden-dim $w \
#                 --reg None \
#                 --lr 0.001 \
#                 --wd $wd \
#                 --epochs 200 \
#                 --log-epoch 10 \
#                 --log-model 10 \
#                 --tag $tag \
#                 --seed $seed \
#                 --eval-sample-size 1000 
#         done
#     done
# done

# multi 
for seed in "401" "402" "403" "404" "405" "501"; do
    for data in "sin-random"; do 
        for w in "8 8 8" "20 20 20"; do
            for wd in "0"; do 
                python src/eval_black_box_robustness.py \
                    --data $data \
                    --step 80 \
                    --hidden-dim $w \
                    --reg None \
                    --lr 0.05 \
                    --wd $wd \
                    --epochs 10000 \
                    --log-epoch 1000 \
                    --log-model 2000 \
                    --tag $tag \
                    --seed $seed \
                    --eval-sample-size 1000 \
                    --eval-epoch 10000 \
                    --vmin -1
            done
        done 
    done
done
