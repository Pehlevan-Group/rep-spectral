#!/bin/bash
#SBATCH -c 2            
#SBATCH -t 0-03:00           
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -o out/baseline_%j.out   
#SBATCH -e out/baseline_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: baseline
# setup parameters
tag='linear_deep_approx'
w=20
reg='None'

for seed in "401" "402" "403" "404" "405" "501"; do
    # for data in "linear" "xor" "sin"; do
    for data in "sin-random"; do
        for wd in "0"; do
            for w in "8  8  8" "20 20 20"; do 
                # # short training
                # python src/train_reg.py \
                #     --data $data \
                #     --hidden-dim $w --reg $reg \
                #     --lr 1 \
                #     --wd $wd --epochs 5000 \
                #     --log-epoch 1000 \
                #     --tag $tag \
                #     --seed $seed
            
                # long training
                python src/train_reg.py \
                    --data $data \
                    --step 80 \
                    --hidden-dim $w \
                    --reg $reg \
                    --lr 0.05 \
                    --wd $wd \
                    --epochs 10000 \
                    --log-epoch 1000 \
                    --log-model 2000 \
                    --tag $tag \
                    --seed $seed \
                    --batch-size 256
            done
        done
    done
done
