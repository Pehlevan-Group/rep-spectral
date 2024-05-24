#!/bin/bash
#SBATCH -c 2            
#SBATCH -t 0-06:00           
#SBATCH --mem=8000
#SBATCH -o out/baseline_eval_%j.out   
#SBATCH -e out/baseline_eval_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: baseline
# setup parameters
tag='b1_long'
w=20
reg='None'

for seed in "401" "402" "403" "404" "405" "501"; do
    for data in "linear" "xor" "sin"; do
        for wd in "0" "1e-4"; do
            # # short training
            # python src/eval_robustness.py \
            #     --data $data \
            #     --hidden-dim $w --reg $reg \
            #     --lr 1 \
            #     --wd $wd --epochs 5000 \
            #     --tag $tag \
            #     --seed $seed \
            #     --eps-size 0.02
                
            
            # long training
            python src/eval_robustness.py \
                --data $data \
                --hidden-dim $w --reg $reg \
                --lr 0.05 \
                --wd $wd --epochs 20000 \
                --tag $tag \
                --seed $seed \
                --eps-size 0.02 
        done
    done
done
