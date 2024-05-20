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
from model import SLP, MLP
from utils import load_config, get_logging_name

# ======== legacy arguments =======
# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on')
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=[20], type=int, help='the number of hidden units', nargs='+')
parser.add_argument('--model', default="34", type=str, help='the model type for ResNet')

# training
parser.add_argument('--batch-size', default=1024, type=int, help='the batchsize for training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=0., type=float, help='weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default="None", type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=None, type=float, 
    help='the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)')
parser.add_argument("--m", default=None, type=int, help='vol element specific: keep the top m singular values to regularize only')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--reg-freq', default=5, type=int, help='the frequency of imposing regularizations')
parser.add_argument('--reg-freq-update', default=None, type=int,
                    help='the frequency of imposing regularization per parameter update: None means reg every epoch only'
                    )
parser.add_argument('--max-layer', default=None, type=int, 
    help='the number of max layers to pull information from. None means the entire feature map')

# iterative update
parser.add_argument('--iterative', action='store_true', default=False, help='True to turn on iterative method')

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
parser.add_argument('--adv-batch-size', default=16, type=int, help='the number of samples to be batched evaluated at a time')
parser.add_argument('--eval-sample-size', default=2000, type=int, help='the number of samples to be evaluated')
parser.add_argument('--attacker', default='TangentAttack', type=str, help='the type of attack')
parser.add_argument('--T', default=40, type=int, help='max iterations for attack')
parser.add_argument('--tol', default=1e-5, type=float, help='the threshold to stop binary search')
parser.add_argument('--vmin', default=0, type=float, help='the min value of the adversarial guess range')
parser.add_argument('--vmax', default=1, type=float, help='the max value of the adversarial guess range')
parser.add_argument('--perturb-vmax', default=0.5, type=float, help='the perturbation range for high dimension adversarial sample discovery')
parser.add_argument('--perturb-vmin', default=-0.5, type=float, help='the pertrubation range for high dimension adversarial sample discovery')
parser.add_argument('--no-shuffle', default=False, action='store_true', help='true to turn off data shuffling in evaluation sampling')

# ========= retrain downstream evaluations ===============
parser.add_argument("--new-head", default=False, action="store_true", help='true to retrain head using multilogistic regression')

args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modify hidden dims
if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1: args.hidden_dim = args.hidden_dim[0]


# set up summary writer
if 'mnist' == args.data: 
    log_name, base_log_name = get_logging_name(args, 'linear_large')
    input_dim, output_dim = 784, 10
elif "cifar10" == args.data:
    log_name, base_log_name = get_logging_name(args, 'conv')
else:
    log_name, base_log_name = get_logging_name(args, 'linear_small')
    input_dim, output_dim = 2, 2

# load data
def load_data():
    if args.data == 'mnist':
        from data import mnist
        train_set, test_set = mnist(paths['data_dir'], flatten=True)
    elif args.data == 'fashion_mnist':
        from data import fashion_mnist
        train_set, test_set = fashion_mnist(paths['data_dir'], flatten=True)
    elif args.data == 'sin-random':
        from data import load_sin_random, CustomDataset
        X_train, X_test, y_train, y_test = load_sin_random(args.step, args.test_size, args.seed)
        train_set, test_set = CustomDataset(X_train, y_train), CustomDataset(X_test, y_test)
    elif args.data == 'xor_symmetric':
        from data import load_xor_symmetric, CustomDataset
        X_train, X_test, y_train, y_test = load_xor_symmetric()
        train_set, test_set = CustomDataset(X_train, y_train), CustomDataset(X_train, y_train) # repeat
    elif args.data == 'xor_noisy':
        from data import load_xor_noisy, CustomDataset
        X_train, X_test, y_train, y_test = load_xor_noisy(args.step, 0.2, args.seed) # * tune std here
        train_set, test_set = CustomDataset(X_train, y_train), CustomDataset(X_train, y_train) # repeat
    elif args.data == "cifar10":
        from data import cifar10
        train_set, test_set = cifar10(paths['data_dir'])
    else:
        raise NotImplementedError(f'{args.data} not available')
    return train_set, test_set

train_set, test_set = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.no_shuffle)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=not args.no_shuffle)
print(f'{args.data} data loaded')

# get model
def load_model():
    """select model"""
    # select nonlinearity 
    nl = getattr(nn, args.nl)()

    # init model
    if args.data == 'cifar10':
        from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        model = eval(f"ResNet{args.model}")(nl=nl)
    else:
        if isinstance(args.hidden_dim, list):
            model = MLP([input_dim, *args.hidden_dim, output_dim], nl=nl)
        else:
            model = SLP(input_dim=input_dim, width=args.hidden_dim, output_dim=output_dim, nl=nl)
    
    # send to proper device
    model = model.to(device)
    return model 

model = load_model()
model.load_state_dict(
    torch.load(
        os.path.join(paths['model_dir'], base_log_name if args.reg == "None" else log_name, f'model_e{args.eval_epoch}.pt'
    ), map_location=device)
)
model.eval()

# wrap model 
if args.new_head:
    from model import ContrastiveWrap
    model = ContrastiveWrap(model, train_loader, device=device, random_state=args.seed)

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

            # * intercept here, if new head, then would like to see the test performance
            if len(correct_samples) >= args.eval_sample_size and not args.new_head: break
        
    # * display test performance 
    if args.new_head:
        print(f"test acc: {len(correct_samples) / len(test_set):.4f}")
    
    # keep first eval_sample_size many samples    
    correct_samples = correct_samples[:args.eval_sample_size]
    if len(target_samples) == 0: target_samples = None # for untargeted attack, change to None
    return correct_samples, target_samples

@torch.no_grad()
def attack_all(samples: torch.Tensor, target_samples: torch.Tensor) -> List[float]:
    """
    perform adversarial attack on correctly predicted samples
    - targeted: pick a random sample of the target class as our initial guess
    - untargeted: pass None and randomized target

    :param samples: batched samples
    :param target_samples: batch samples of target class
    :param return l2 adversarial distance
    """
    attacker = Attacker(
        model, samples, target_samples, 
        args.adv_batch_size,
        args.tol, 
        vmin=args.vmin, vmax=args.vmax, T=args.T,
        device=device
    )
    dists, _ = attacker.attack()
    return dists.tolist()

def record(dists: List[float]):
    """dump computed distance to files"""
    series = pd.Series(dists)
    csv_log_name = f'black_box_l2_e{args.eval_epoch}_{args.attacker}_tar{args.target}_T{args.T}_tol{args.tol}'
    if args.new_head:
        csv_log_name += "_newhead"
    series.to_csv(
        os.path.join(paths['result_dir'],  base_log_name if args.reg == "None" else log_name, 
        f'{csv_log_name}.csv')
    )

@torch.no_grad()
def main():
    samples, target_samples = get_samples()
    dists = attack_all(samples, target_samples)
    record(dists)

if __name__ == '__main__':
    main()
