# parameters
epochs=15000
burnin=10500
log_model=1500
log_epoch=750

lr=1
wd=1e-4 # ! tune weight decay here 
nl="GELU"
lam=1e-3

# # noisy data
# tag="xor_noisy_gelu2"
# data="xor_noisy"
# bs=20
# eval_samples=80

# clean xor 
tag="xor_gelu"
data="xor_symmetric"
bs=4
eval_samples=4

with_new_head=""
with_new_head="--new-head"

# start training
for seed in "400" "500" "600" "700" "800"; do
    for hd in "3" "8"; do 
        # base model
        python src/eval_black_box_robustness.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed \
            --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head

        # reg model
        python src/eval_black_box_robustness.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed --iterative \
             --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head
        
        # spectral model
        python src/eval_black_box_robustness.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed --iterative \
             --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head
        
        # reg model
        python src/eval_black_box_robustness.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed \
             --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head
        
        # spectral model
        python src/eval_black_box_robustness.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed \
            --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head
    done
done

# seed=400
# hd=8
# python src/eval_black_box_robustness.py \
#     --data $data --batch-size $bs \
#     --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin 10500 \
#     --epochs $epochs --tag $tag \
#     --log-epoch $log_epoch --log-model $log_model \
#     --seed $seed --iterative \
#     --eval-epoch 15000 --eval-sample-size $eval_samples --vmin -2 --vmax 2 --no-shuffle $with_new_head --new-head
