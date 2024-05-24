#!/bin/bash
#SBATCH -c 2               
#SBATCH -t 1-00:00           
#SBATCH --gres=gpu:1
#SBATCH --mem=8000   
#SBATCH -o out/reg_%j.out   
#SBATCH -e out/reg_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: regularize
# set up parameters
tag='linear_deep'
w=8
reg='spectral'

# single-hidden-layer
# for seed in "401" "402" "403" "404" "405" "501"; do
#     # for data in "linear" "xor" "sin"; do
#     for data in "sin"; do
#         # for lam in "1e-4" "1e-3" "1e-2" "1e-1"; do
#         for lam in "1e-5" "1e-4" "4e-3" "1e-3"; do
#             for ss in "0.1" "0.2" "0.3"; do
#                 for wd in "0" "1e-4"; do
#                     # # short training
#                     # python src/train_reg.py \
#                     #     --data $data --lam $lam \
#                     #     --hidden-dim $w --reg $reg \
#                     #     --lr 1 --sample-size $ss \
#                     #     --wd $wd --epochs 5000 \
#                     #     --burnin 2500 \
#                     #     --log-epoch 1000 \
#                     #     --tag $tag \
#                     #     --seed $seed
                    
#                     # long training
#                     python src/train_reg.py \
#                         --data $data --lam $lam \
#                         --hidden-dim $w --reg $reg \
#                         --lr 0.05 --sample-size $ss \
#                         --wd $wd --epochs 20000 \
#                         --burnin 10000 \
#                         --log-epoch 1000 \
#                         --tag $tag \
#                         --seed $seed
#                 done
#             done
#         done
#     done
# done


# multi-hidden-layers
for seed in "401" "402" "403" "404" "405" "501"; do
    for lam in "1e-3" "1e-4" "1e-5"; do
        for w in "8 8 8" "20 20 20"; do
            if [ $reg == "spectral" -o $reg == "cross-lip" ]
            then
                python src/train_reg.py \
                    --data sin-random \
                    --step 80 \
                    --lam $lam \
                    --hidden-dim $w \
                    --reg $reg \
                    --wd 0 \
                    --epochs 10000 \
                    --tag $tag \
                    --log-epoch 1000 \
                    --log-model 2000 \
                    --seed $seed \
                    --lr 0.05 \
                    --test-size 0.5 \
                    --burnin 0 \
                    --sample-size 0.2 \
                    --batch-size 256 # \
                    # --iterative # ! power method
            else
                for max_layer in "1" "2" "3"; do
                    python src/train_reg.py \
                        --data sin-random \
                        --step 80 \
                        --lam $lam \
                        --hidden-dim $w \
                        --reg $reg \
                        --wd 0 \
                        --epochs 10000 \
                        --tag $tag \
                        --log-epoch 1000 \
                        --log-model 2000 \
                        --seed $seed \
                        --lr 0.05 \
                        --test-size 0.5 \
                        --burnin 0 \
                        --sample-size 0.2 \
                        --max-layer $max_layer \
                        --batch-size 256 # \
                        # --iterative # ! power method
                done
            fi
        done
    done
done
