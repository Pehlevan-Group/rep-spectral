"""
contrastive evaluation of black box robustness
also used for simple transfer learning task
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
from model import BarlowTwins, SimCLR, ContrastiveWrap, get_contrastive_acc
from utils import load_config, get_logging_name

# ======== legacy arguments =======
# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on')
parser.add_argument("--target-data", default='cifar10', type=str, 
    help='the data to be transferred to (for contrastive this is the same as training set, transfer otherwise)')
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=[20], type=int, help='the number of hidden units', nargs='+')
parser.add_argument('--model', default="34", type=str, help='the model type for ResNet')
parser.add_argument('--model-type', default=None, type=str, help='contrastive model type')

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

# barlow specific 
parser.add_argument("--projector", default=[8192, 8192, 8192], type=int, nargs='+', help='the projector MLP dimensions')
parser.add_argument("--lambd", default=0.005, type=float, help='weight on off-diagonal terms')

# SimCLR specific
parser.add_argument("--temperature", default=0.07, type=float, help="the temperature in softmax")

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

args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modify hidden dims
if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1: args.hidden_dim = args.hidden_dim[0]


# set up summary writer
log_name, base_log_name = get_logging_name(args, args.model_type)

# load data
def load_data():
    if args.target_data == "cifar10":
        # for barlow and others, data are normalized, use normalize cifar10
        if args.model_type == 'barlow' or args.model_type == 'transfer':
            from data import cifar10
            train_set, test_set = cifar10(paths['data_dir'])
        elif args.model_type == 'simclr':
            from data import cifar10_clean
            train_set, test_set = cifar10_clean(paths["data_dir"])
        else:
            raise NotImplementedError(f"undefined model type for evaluation: {args.model_type}")
    else:
        raise NotImplementedError(f'{args.target_data} not available')
    return train_set, test_set

train_set, test_set = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=not args.no_shuffle)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=not args.no_shuffle)
print(f'{args.target_data} data loaded')

# get model
def load_model():
    """select model"""
    # select nonlinearity 
    nl = getattr(nn, args.nl)()

    # init model
    from model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    
    # get number of classes to load checkpoints
    if args.data == 'cifar100':
        num_classes = 100
    elif args.data == 'cifar10':
        num_classes = 10
    else:
        raise NotImplementedError(f"unrecognized training set {args.data}")
    
    backbone = eval(f"ResNet{args.model}")(num_classes=num_classes, nl=nl).to(device)
    if args.model_type == "barlow":
        model_base = BarlowTwins(backbone, args.batch_size, args.projector, args.lambd, nl=nl).to(device)
    elif args.model_type == 'simclr':
        model_base = SimCLR(backbone, args.batch_size, nl=nl).to(device)
    elif args.model_type == 'transfer':
        model_base = backbone
    else:
        raise NotImplementedError(f"contrastive model type {args.model_type} not available")
    
    # load trained parameters
    model_base.load_state_dict(
        torch.load(
            os.path.join(paths['model_dir'], base_log_name if args.reg == "None" else log_name, f'model_e{args.eval_epoch}.pt'
                        ), map_location=device)
    )
    model_base.eval()

    # wrap feature map + logistic into an nn module 
    model = ContrastiveWrap(model_base, train_loader, device=device, random_state=args.seed)
    model = model.to(device)
    return model 

model = load_model()


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
    series.to_csv(
        os.path.join(paths['result_dir'],  base_log_name if args.reg == "None" else log_name, 
        f'black_box_l2_e{args.eval_epoch}_{args.attacker}_tar{args.target}_T{args.T}_tol{args.tol}.csv')
    )

@torch.no_grad()
def main():
    # for pretrain set different from eval set, we perform an evaluation
    if args.data != args.target_data:
        # get train and test labels
        train_labels, test_labels = [], []
        for _, labels in train_loader:
            train_labels.append(labels)
        for _, labels in test_loader:
            test_labels.append(labels)

        train_labels = torch.concat(train_labels, dim=0)
        test_labels = torch.concat(test_labels, dim=0)

        downstream_train_acc, downstream_test_acc = get_contrastive_acc(
            model, 
            train_loader, train_labels, test_loader, test_labels, 
            device=device, random_state=args.seed
        )
        print(f"downstream train acc: {downstream_train_acc:.4f}; downstream test acc: {downstream_test_acc:.4f}")

    samples, target_samples = get_samples()
    dists = attack_all(samples, target_samples)
    record(dists)

if __name__ == '__main__':
    main()
