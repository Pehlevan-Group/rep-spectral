# submit mnist training jobs 
seeds="400 500 600 700 800"
for wd in "0" "1e-4"; do 
    for lam in "0.001" "0.01" "0.1"; do 
        sbatch commands/train_mnist.sh "$seeds" "$wd" "$lam"
    done 
done 

seeds="40 50 60 70 80"
for wd in "0" "1e-4"; do 
    for lam in "0.001" "0.01" "0.1"; do 
        sbatch commands/train_mnist.sh "$seeds" "$wd" "$lam"
    done 
done 

# # submit mnist training jobs (for vis)
# seeds="400 500 600 700 800"
# for wd in "0" "1e-4"; do 
#     for lam in "0.001" "0.01" "0.1"; do 
#         sbatch commands/compute_mnist.sh "$seeds" "$wd" "$lam"
#     done 
# done 

# seeds="40 50 60 70 80"
# for wd in "0" "1e-4"; do 
#     for lam in "0.001" "0.01" "0.1"; do 
#         sbatch commands/compute_mnist.sh "$seeds" "$wd" "$lam"
#     done 
# done 
