# setup parameters
tag='b1'
w=20
reg='None'

for seed in "401" "402" "403" "404" "405" "501"; do
    for data in "linear" "xor" "sin"; do
        for wd in "0" "1e-4"; do
            # short training
            python src/train_reg.py \
                --data $data \
                --hidden-dim $w --reg $reg \
                --lr 1 \
                --wd $wd --epochs 5000 \
                --tag $tag \
                --seed $seed
            
            # long training
            python src/train_reg.py \
                --data $data \
                --hidden-dim $w --reg $reg \
                --lr 1 \
                --wd $wd --epochs 20000 \
                --tag $tag \
                --seed $seed
        done
    done
done


python src/eval_robustness.py --data xor --hidden-dim 8 --reg None --lr 1 --wd 1e-4 --epochs 5000 --tag b1 --seed 401

python src/eval_black_box_robustness.py --data mnist --hidden-dim 2000 --reg None --lr 0.001 --wd 0. --epochs 200 --tag mnist --seed 401 --target 3 --eval-sample-size 200 
