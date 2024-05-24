#!/bin/bash
#SBATCH --job-name=transfer_base
#SBATCH --nodes=1            # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH -t 0-06:00
#SBATCH --mem=600GB
#SBATCH --array=0-9%10
#SBATCH -o out/transfer/transfer_%A_%a.out
#SBATCH -e out/transfer/transfer_%A_%a.err
#SBATCH --mail-type=END,FAIL

# activate env
source ~/.bashrc
conda activate curvature

# train resnet
model='34'      # width of intermediate layer 
batchsize=128
epochs=200
data="cifar100"
log_epoch=5
log_model=20

# get seed and lam
seed=$((SLURM_ARRAY_TASK_ID % 10 + 400))

# testing learning rates
lr="0.1"
wd=5e-4

# specify paths
tag='transfer'

# reg setup 
lam=0.01
burnin=160

train_modifier="--reg-freq-update 1"
# train_modifier="--max-layer 2"
# train_modifier="--reg-freq-update 15"
# train_modifier="--reg-freq-update 24 --max-layer 2"
# train_modifier=""

python src/train_reg_transfer.py --seed $seed --model $model --epochs $epochs --lr $lr --wd $wd \
 --data $data --batch-size $batchsize --log-epoch $log_epoch --log-model $log_model --burnin $burnin \
 --lam $lam --reg None --tag $tag $train_modifier

# TODO: setup evaluation
python src/eval_black_box_robustness_contrastive.py \
--model $model --model-type transfer --epochs $epochs \
--seed $seed --lr $lr --wd $wd --nl GELU --data $data \
--batch-size $batchsize --tag $tag \
--log-model $log_model --log-epoch $log_epoch --lam $lam --reg None --burnin $burnin \
--vmin -3 --vmax 3 --perturb-vmin -0.3 --perturb-vmax 0.3 \
--eval-epoch $epochs --eval-sample-size 1000 --reg-freq 1 \
$train_modifier
