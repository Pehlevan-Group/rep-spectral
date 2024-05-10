"""
transfer learning evaluation of black box robustness
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
from model import ResNet50Pretrained, TransferWrap
from utils import load_config, get_logging_name

# ======== legacy arguments =======
# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='dog', type=str, help='data to perform training on')

# model
parser.add_argument('--model', default='50', type=str, help="resnet model number", choices=["18", "34", "50", "101", "152"])

# training
parser.add_argument('--batch-size', default=64, type=int, help='the batchsize for training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for the last layer')
parser.add_argument("--nl", default='GELU', type=str, help='the nonlinearity throughout')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=0., type=float, help='the weight decay during model training')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--alpha', default=0.1, type=float, help='strength of l2sp')
parser.add_argument('--beta', default=0.01, type=float, help='strength for l2-norm new head in transfer architecture')
parser.add_argument('--lam', default=1e-4, type=float, help='the multiplier / strength of regularization')
parser.add_argument("--max-layer", default=None, type=int, 
                    help='the number of layers to regularize in ResNet architecture; None to regularize all'
                    )
parser.add_argument('--reg', default='None', type=str, help='the type of regularization', nargs='+')
parser.add_argument("--epochs", default=200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=0, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--reg-freq', default=1, type=int, help='the frequency of imposing regularizations')
parser.add_argument('--reg-freq-update', default=None, type=int, 
                    help='the frequency of imposing convolution singular value regularization per parameter update: None means reg every epoch only'
                    )
parser.add_argument("--custom-pretrain", default=False, action='store_true', help='True to turn on custom pretrain')

# # iterative singular 
parser.add_argument('--iterative', action='store_true', default=False, help='True to turn on iterative method')
parser.add_argument('--eps', default=1e-4, type=float, help='the tolerance for stopping the iterative method')
parser.add_argument('--max-update', default=1, type=int, help='the maximum update iteration during training')

# logging
parser.add_argument('--log-epoch', default=5, type=int, help='logging frequency')
parser.add_argument('--log-model', default=20, type=int, help='model logging frequency')

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

# parameter compatibility check
if args.reg_freq_update is not None: 
    assert args.reg_freq == 1, f"reg_freq_update{args.reg_freq_update} & reg_freq{args.reg_freq} not compatible"

# set up summary writer
log_name, base_log_name = get_logging_name(args, 'finetune')

# load data
def load_data():
    if args.data == 'dog':
        from data import load_dog
        train_set, test_set = load_dog(paths['data_dir'])
        num_classes = 120
    elif args.data == 'flower':
        from data import load_flower 
        train_set, test_set = load_flower(paths['data_dir'])
        num_classes = 102
    elif args.data == 'indoor':
        from data import load_indoor 
        train_set, test_set = load_indoor(paths['data_dir'])
        num_classes = 67
    elif args.data == 'cifar10':
        from data import cifar10
        train_set, test_set = cifar10(paths['data_dir'])
        num_classes = 10
    elif args.data == 'cifar10_resized':
        from data import cifar10_resized
        train_set, test_set = cifar10_resized(paths['data_dir'])
        num_classes = 10
    else:
        raise NotImplementedError(f"{args.data} is not available")
    return train_set, test_set, num_classes

train_set, test_set, num_classes = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
print(f'{args.data} data loaded')

# get model
def load_model():
    if args.custom_pretrain:
        from model import ResNet50
        model_base = ResNet50(num_classes, small_input=args.data=='cifar10').to(device)
    else:
        model_base = ResNet50Pretrained(
            num_classes,
            small_conv1=args.data=='cifar10' # change 'conv1' layer for small dimensional data
        ).to(device)

    model_base.load_state_dict(
        torch.load(
            os.path.join(paths["model_dir"], base_log_name if "None" in args.reg else log_name, f"model_e{args.eval_epoch}.pt"),
            map_location=device
        )
    )
    model_base.eval()
    if args.custom_pretrain: return model_base
    model = TransferWrap(model_base) # discard feature representations while calling forward
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
        os.path.join(paths['result_dir'],  base_log_name if "None" in args.reg else log_name, 
        f'black_box_l2_e{args.eval_epoch}_{args.attacker}_tar{args.target}_T{args.T}_tol{args.tol}.csv')
    )

@torch.no_grad()
def main():
    samples, target_samples = get_samples()
    dists = attack_all(samples, target_samples)
    record(dists)

if __name__ == '__main__':
    main()
