#!/bin/bash
#SBATCH -c 2               
#SBATCH -t 0-06:00           
#SBATCH --mem=8000   
#SBATCH -o out/reg_eval_%j.out   
#SBATCH -e out/reg_eval_%j.err 
#SBATCH --mail-type=END,FAIL

# initial training: regularize
# set up parameters
tag='b1_long'
w=20
reg='vol'

for seed in "401" "402" "403" "404" "405" "501"; do
    # for data in "linear" "xor" "sin"; do
    for data in "sin"; do
        # for lam in "1e-4" "1e-3" "1e-2" "1e-1"; do
        for lam in "1e-5" "1e-4" "4e-3" "1e-3"; do
            for ss in "0.1" "0.2" "0.3"; do
                for wd in "0" "1e-4"; do
                    # # short training
                    # python src/eval_robustness.py \
                    #     --data $data --lam $lam \
                    #     --hidden-dim $w --reg $reg \
                    #     --lr 1 --sample-size $ss \
                    #     --wd $wd --epochs 5000 \
                    #     --burnin 2500 \
                    #     --tag $tag \
                    #     --seed $seed \
                    #     --eps-size 0.02 
                    
                    # long training
                    python src/eval_robustness.py \
                        --data $data --lam $lam \
                        --hidden-dim $w --reg $reg \
                        --lr 0.05 --sample-size $ss \
                        --wd $wd --epochs 20000 \
                        --burnin 10000 \
                        --tag $tag \
                        --seed $seed \
                        --eps-size 0.02
                done
            done
        done
    done
done
