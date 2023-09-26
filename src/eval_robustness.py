"""
evaluate robustness of a model by producing 
perturbation vs acc plot
"""

# load packages
import os
import argparse
from typing import Iterable
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# load files
from utils import load_config
from model import SLP
from adversarial import pgd_perturbation

parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on', choices=['linear', 'xor', 'sin'])
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=20, type=int, help='the number of hidden units')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument("--wd", default=0, type=float, help='weight decay')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default=None, type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=None, type=float, 
    help='the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')

# adv
parser.add_argument('--max-eps', default=1, type=float, help='the maximum perturbation level')
parser.add_argument('--eps-size', default=0.1, type=float, help='the step size in increasing eps')
parser.add_argument('--norm', default=2, type=str, help='the norm of perturbation')
parser.add_argument('--k', default=10, type=int, help='the number of steps to make white box attacks')
parser.add_argument('--step-size', default=0.1, type=float, help='the size to make perturbation at each step')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')
args = parser.parse_args()

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up summary writer
log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}' + \
           f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam{args.lam}_reg{args.reg}' + \
           f'_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}'
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)

# load data
def load_data():
    if args.data == 'linear':
        from data import load_linear_boundary
        X_train, X_test, y_train, y_test = load_linear_boundary(args.step, args.test_size, args.seed)
    elif args.data == 'xor':
        from data import load_xor_boundary
        X_train, X_test, y_train, y_test = load_xor_boundary(args.step, args.test_size, args.seed)
    elif args.data == 'sin':
        from data import load_sin_boundary
        X_train, X_test, y_train, y_test = load_sin_boundary(args.step, args.test_size, args.seed)
    else:
        raise NotImplementedError(f'{args.data} not available')
    return X_train, X_test, y_train, y_test

# keep test data only
_, X_test, _, y_test = load_data()
X_test, y_test = X_test.to(device), y_test.to(device)
print(f'{args.data} data loaded')

# get model
nl = getattr(nn, args.nl)()
model = SLP(width=args.hidden_dim, nl=nl).to(device)

# read model 
model.load_state_dict(torch.load(os.path.join(paths['model_dir'], log_name, 'model.pt'), map_location=device))
print("model loaded")

def get_adv_acc():
    """compute adversarial accuracy"""
    # filter on correctly identified samples
    correct_idx = (model(X_test).flatten() > 0).to(torch.int32) == y_test
    X_test_samples, y_test_samples = X_test[correct_idx], y_test[correct_idx]
    # loop through each max epsilon to get adversarial accuracy
    acc_list = [1]
    loss_fn, sig = nn.BCELoss(), nn.Sigmoid()
    t = int(args.max_eps / args.eps_size)
    for i in tqdm(range(1, t + 1)):
        X_test_perturbed = pgd_perturbation(
            model, X_test_samples, y_test_samples, 
            lambda y1, y2: loss_fn(sig(y1).flatten(), y2), 
            args.norm,
            args.k,
            args.step_size,
            i * args.eps_size
        )

        # get new acc 
        acc = ((model(X_test_perturbed).flatten() > 0).to(torch.int32) == y_test_samples).to(torch.float32).mean().item()
        acc_list.append(acc)

    return acc_list

def log_adv_acc(acc_list: Iterable):
    """record evaluation metrics"""
    s = pd.DataFrame(
        [np.linspace(0, args.max_eps, int(args.max_eps / args.eps_size) + 1), acc_list],
        index=['eps','acc']
    ).T
    s.to_csv(os.path.join(paths['result_dir'], log_name, f'adv_acc_n{args.norm}_k{args.k}_ss{args.step_size}.csv'))

def main():
    acc_list = get_adv_acc()
    log_adv_acc(acc_list)

if __name__ == '__main__':
    main()