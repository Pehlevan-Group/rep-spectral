"""
Evaluate Black-Box Robustness
"""

# load packages
import os
import argparse
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# load files
from model import SLP
from utils import load_config

# ======== legacy arguments =======
# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on')
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=20, type=int, help='the number of hidden units')

# training
parser.add_argument('--batch-size', default=1024, type=int, help='the batchsize for training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument("--wd", default=0., type=float, help='weight decay')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default="None", type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=None, type=float, 
    help='the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)')
parser.add_argument("--m", default=None, type=int, help='vol element specific: keep the top m singular values to regularize only')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--reg-freq', default=5, type=int, help='the freqency of imposing regularizations')

# logging
parser.add_argument('--log-epoch', default=100, type=int, help='logging frequency')
parser.add_argument('--log-model', default=10, type=int, help='model logging frequency')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')
parser.add_argument('--scanbatchsize', default=20, type=int, help='the number of samples to batch compute jacobian for')

# ========= evaluation arguments =======
parser.add_argument('--eval-epoch', default=200, type=int, help='the epoch of model to be evaluated at')
parser.add_argument('--target', default=None, type=int, help='the target adversarial class')
parser.add_argument('--eval-sample-size', default=2000, type=int, help='the number of samples to be evaluated')
parser.add_argument('--attacker', default='TangentAttack', type=str, help='the type of attack')
parser.add_argument('--T', default=40, type=int, help='max iterations for attack')
parser.add_argument('--tol', default=1e-5, type=float, help='the threshold to stop binary search')

args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up summary writer
log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}' + \
           f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam{args.lam}_reg{args.reg}' + (f'_m{args.m}' if args.reg == 'vol' else '') + \
           f'_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}'
base_log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}' + \
           f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam1_regNone' + \
           f'_ssNone_e{args.epochs}_b600_seed{args.seed}'
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)

# load data
def load_data():
    if args.data == 'mnist':
        from data import mnist
        train_set, test_set = mnist(paths['data_dir'], flatten=True)
    elif args.data == 'fashion_mnist':
        from data import fashion_mnist
        train_set, test_set = fashion_mnist(paths['data_dir'], flatten=True)
    else:
        raise NotImplementedError(f'{args.data} not available')
    return train_set, test_set

train_set, test_set = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
print(f'{args.data} data loaded')

# get model
nl = getattr(nn, args.nl)()
model = SLP(input_dim=784, width=args.hidden_dim, output_dim=10, nl=nl).to(device)
model.load_state_dict(
    torch.load(
        os.path.join(paths['model_dir'], base_log_name if args.reg == "None" else log_name, f'model_e{args.eval_epoch}.pt'
    ), map_location=device)
)

# get attacker
def get_attacker():
    if args.attacker == 'TangentAttack':
        from adversarial import TangentAttack
        Attacker = TangentAttack
    else:
        raise NotImplementedError(f'attacker {args.attacker} not implemented')
    return Attacker

Attacker = get_attacker()

# get samples
@torch.no_grad()
def get_samples() -> Tuple[torch.Tensor]:
    """
    filter on test sample with correct predictions

    :return correctly predicted samples, along with None/target samples
    """
    correct_samples, target_samples = torch.tensor([]), torch.tensor([])
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X).argmax(dim=-1)

        if args.target is not None:
            # collect adversarial target samples (using the true label, and equals the prediction)
            target_samples = torch.cat([target_samples, X[(y == args.target) & (y_pred.eq(y))].to('cpu')])
            # collect correct samples and samples that do not belong to the adversarial class 
            correct_samples = torch.cat([correct_samples, X[(y != args.target) & (y_pred.eq(y))].to('cpu')])
        else:
            correct_samples = torch.cat([correct_samples, X[y_pred.eq(y)].to('cpu')])
            if len(correct_samples) >= args.eval_sample_size: break

    # keep first eval_sample_size many samples    
    correct_samples = correct_samples[:args.eval_sample_size]
    if len(target_samples) == 0: target_samples = None # for untargeted attack, chenge to None
    return correct_samples, target_samples

@torch.no_grad()
def attack_all(samples: torch.Tensor, target_samples: torch.Tensor) -> List[float]:
    """
    perform adversarial attck on correctly predicted samples
    - targeted: pick a random sample of the target class as our intial guess
    - untargeted: pass None and randomized target

    :param samples: batched samples
    :param target_samples: batch samples of target class
    :param return l2 adversarial distance
    """
    dists = []
    for sample in tqdm(samples):
        # keep first dimension the batch dimension
        sample = sample.unsqueeze(dim=0).to(device)
        
        # targeted: draw a random one 
        if target_samples is not None:
            target_sample = target_samples[torch.randperm(len(target_samples))[[1]]]
            target_sample = target_sample.to(device)
        # untargeted
        else:
            target_sample = None
            
        attacker = Attacker(model, sample, target_sample, args.tol, vmin=0, vmax=1, T=args.T)
        perturbed_sample = attacker.attack()

        # get distance
        dists.append((perturbed_sample - sample).norm(p=2).item())
    
    return dists

def record(dists: List[float]):
    """dump computed distance to files"""
    series = pd.Series(dists)
    series.to_csv(
        os.path.join(paths['result_dir'],  base_log_name if args.reg == "None" else log_name, 
        f'black_box_l2_e{args.eval_epoch}_{args.attacker}_tar{args.target}_T{args.T}_tol{args.tol}.csv')
    )

@torch.no_grad()
def main():
    samples, target_samples = get_samples()
    dists = attack_all(samples, target_samples)
    record(dists)

if __name__ == '__main__':
    main()