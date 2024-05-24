# batch job submit 
# global settings
lam=0.005

# seed setting 1 
seeds="400 500 600 700 800"
for wd in "0" "1e-4"; do 
    # # base 
    # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "None" "$lam" ""

    # for reg in "spectral" "eig-ub"; do 
    for reg in "spectral"; do
        # # no modifier
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "" 

        # # only two layers
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--max-layer 2"

        # more regularization
        sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--reg-freq-update 24"

        # # regularization and two layers
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--reg-freq-update 24 --max-layer 2"
    done 
done 


# seed setting 2
seeds="40 50 60 70 80"
for wd in "0" "1e-4"; do 
    # # base 
    # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "None" "$lam" ""
    
    # for reg in "spectral" "eig-ub"; do 
    for reg in "spectral"; do
        # # no modifier
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" ""

        # # only two layers
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--max-layer 2"

        # more regularization
        sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--reg-freq-update 24"

        # # regularization and two layers
        # sbatch commands/job_train_resnet.sh "$seeds" "$wd" "$reg" "$lam" "--reg-freq-update 24 --max-layer 2"
    done 
done 
