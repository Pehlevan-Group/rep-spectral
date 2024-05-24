#!/bin/bash
#SBATCH -c 2            
#SBATCH -t 0-08:00           
#SBATCH --mem=20000
#SBATCH -o out/baseline_eval_bb_%j.out   
#SBATCH -e out/baseline_eval_bb_%j.err 
#SBATCH --mail-type=END,FAIL

data="mnist"
tag="mnist_approx"
wd=0
reg="eig-ub"

# single-hidden-layer
for seed in "401" "402" "403" "404" "405" "501"; do
# for seed in "403" "404" "405" "501"; do
    # for ss in "0.1" "0.2"; do 
    for ss in "0.2"; do
        for burnin in "0"; do
            python src/eval_black_box_robustness.py \
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
                --eval-sample-size 3000 
            # done 

            python src/eval_black_box_robustness.py \
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
                --eval-sample-size 3000 
        done
    done 
done 

# tag='linear_deep_approx'
# w=8
# reg='eig-ub'

# # multi-hidden layer
# # for seed in "401" "402" "403" "404" "405" "501"; do
# for seed in "405" "501"; do
#     for lam in "1e-3" "1e-4" "1e-5"; do
#         for w in "8 8 8" "20 20 20"; do
#             if [ $reg == "spectral" -o $reg == "cross-lip" ]
#             then
#                 python src/eval_black_box_robustness.py \
#                     --data sin-random \
#                     --step 80 \
#                     --lam $lam \
#                     --hidden-dim $w \
#                     --reg $reg \
#                     --wd 0 \
#                     --epochs 10000 \
#                     --tag $tag \
#                     --log-epoch 1000 \
#                     --log-model 2000 \
#                     --seed $seed \
#                     --lr 0.05 \
#                     --test-size 0.5 \
#                     --burnin 0 \
#                     --sample-size 0.2 \
#                     --eval-epoch 10000 \
#                     --eval-sample-size 1000 \
#                     --vmin -1
#             else
#                 for max_layer in "1" "2" "3"; do
#                     python src/eval_black_box_robustness.py \
#                         --data sin-random \
#                         --step 80 \
#                         --lam $lam \
#                         --hidden-dim $w \
#                         --reg $reg \
#                         --wd 0 \
#                         --epochs 10000 \
#                         --tag $tag \
#                         --log-epoch 1000 \
#                         --log-model 2000 \
#                         --seed $seed \
#                         --lr 0.05 \
#                         --test-size 0.5 \
#                         --burnin 0 \
#                         --sample-size 0.2 \
#                         --max-layer $max_layer \
#                         --eval-epoch 10000 \
#                         --eval-sample-size 1000 \
#                         --vmin -1
#                 done
#             fi
#         done
#     done
# done