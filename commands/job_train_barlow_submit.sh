# tune regularization
lam=0.1

# submit barlow base jobs 
# for seed in "400" "401" "402" "403" "404" "405" "406" "407" "408" "409"; do
for seed in "400" "401" "402" "403" "404"; do
# for seed in "400" "401"; do
# for seed in "400" "401" "402" "403" "404" "405" "406" "407" "408"; do
# for seed in "409"; do 
    for wd in "0" "1e-4"; do 
    # for wd in "1e-4"; do 
        for reg in "None"; do 
            sbatch commands/job_train_barlow.sh "$seed" "$wd" "$reg" "$lam" ""
        done 

        # for reg in "eig-ub"; do 
        #     # update freq 24
        #     sbatch commands/job_train_barlow.sh "$seed" "$wd" "$reg" "$lam" "--reg-freq-update 24"

        #     # update freq 10 
        #     sbatch commands/job_train_barlow.sh "$seed" "$wd" "$reg" "$lam" "--reg-freq-update 10"
        # done 
    done
done
