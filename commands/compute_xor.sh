# parameters
epochs=15000
burnin=10500
log_model=1500
log_epoch=750

lr=1
wd=1e-4 # ! tune weight decay here 
nl="GELU"
lam=1e-3

# noisy data
tag="xor_noisy_gelu2"
data="xor_noisy"
bs=20

# # clean data
# tag="xor_gelu"
# data="xor_symmetric"
# bs=4

# element computation
sample_step=40
scan_batchsize=512

# start training
# for seed in "400" "500" "600" "700" "800"; do
for seed in "400"; do
# for seed in "600"; do
    # for hd in "3" "8"; do
    for hd in "8"; do 
        # base model
        python src/compute_reg.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg None \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed --sample-step $sample_step --scan-batchsize $scan_batchsize

        # reg model
        python src/compute_reg.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed --iterative --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # spectral model
        python src/compute_reg.py \
            --data $data --batch-size $bs \
            --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin 10500 \
            --epochs $epochs --tag $tag \
            --log-epoch $log_epoch --log-model $log_model \
            --seed $seed --iterative --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # # reg model
        # python src/compute_reg.py \
        #     --data $data --batch-size $bs \
        #     --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg eig-ub --lam $lam --burnin 10500 \
        #     --epochs $epochs --tag $tag \
        #     --log-epoch $log_epoch --log-model $log_model \
        #     --seed $seed --sample-step $sample_step --scan-batchsize $scan_batchsize
        
        # # spectral model
        # python src/compute_reg.py \
        #     --data $data --batch-size $bs \
        #     --hidden-dim $hd --nl $nl --lr $lr --wd $wd --reg spectral --lam $lam --burnin 10500 \
        #     --epochs $epochs --tag $tag \
        #     --log-epoch $log_epoch --log-model $log_model \
        #     --seed $seed --sample-step $sample_step --scan-batchsize $scan_batchsize
    done
done 
