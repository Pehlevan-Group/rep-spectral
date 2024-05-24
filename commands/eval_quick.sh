#!/bin/bash
#SBATCH -c 10
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o out/quick/quick_%j.out
#SBATCH -e out/quick/quick_%j.err
#SBATCH --mail-type=END,FAIL

python src/eval_black_box_quick.py
