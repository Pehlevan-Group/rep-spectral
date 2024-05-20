"""
compute the top eigenvalue at sample points and store for 2D sample points 
"""

# load packages
import os
import argparse
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# load file
from data import CustomScan
from model import SLP, MLP
from utils import load_config, get_logging_name, determinant_analytic, top_eig_analytic

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on')
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')
parser.add_argument('--batch-size', default=512, type=int, help='the training batch size')

# model
parser.add_argument('--hidden-dim', default=[20], type=int, help='the number of hidden units', nargs='+')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=0, type=float, help='weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default='None', type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=None, type=float, 
    help='the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--max-layer', default=None, type=int, 
    help='the number of max layers to pull information from. None means the entire feature map')

# iterative singular 
parser.add_argument('--iterative', action='store_true', default=False, help='True to turn on iterative method')
parser.add_argument('--tol', default=1e-6, type=float, help='the tolerance for stopping the iterative method')
parser.add_argument('--max-update', default=10, type=int, help='the maximum update iteration during training')

# logging
parser.add_argument('--log-epoch', default=100, type=int, help='logging frequency')
parser.add_argument('--log-model', default=10, type=int, help='model logging frequency')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')

# plotting the sample points
parser.add_argument("--sample-step", default=40, type=int, help='the number of samples along each axis for computing')
parser.add_argument("--scan-batchsize", default=16, type=int, help='the number of samples to batch for computations')

args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load sample points (for XOR dataset)
def load_sample_points(low=-3.5, high=3.5):
    """generate 2D uniform sample points from within unit low high"""
    samples_1d = torch.linspace(low, high, args.sample_step)
    samples = torch.cartesian_prod(samples_1d, samples_1d)
    # wrap to a dataset 
    dataset = CustomScan(samples)
    return dataset

def load_model() -> nn.Module:
    """get base model from config"""
    nl = getattr(nn, args.nl)()
    if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1:
            # single-hidden-layer
        args.hidden_dim = args.hidden_dim[0]
        model = SLP(input_dim=2, width=args.hidden_dim,
                    output_dim=2, nl=nl).to(device)
    else:
        # multi-hidden-layer
        model = MLP([2, *args.hidden_dim, 2], nl=nl).to(device)
    return model


def main():
    # get samples
    scan_dataset = load_sample_points()
    loader = DataLoader(scan_dataset, args.scan_batchsize, shuffle=False, drop_last=False)
    
    # get model 
    model = load_model()

    # set up summary writer
    log_name, base_log_name = get_logging_name(args, 'linear_small')
    if args.reg == 'None': log_name = base_log_name
    result_path = os.path.join(paths['result_dir'], log_name)

    # compute quantities 
    # initialize
    if args.reg == 'None':
        pbar = tqdm(range(0, args.epochs + 1, args.log_model))
    else:
        # start training from burnin
        pbar = tqdm(range(args.burnin, args.epochs + 1, args.log_model))
    
    for i in pbar:
        # load model
        model.load_state_dict(torch.load(os.path.join(paths['model_dir'], log_name, f"model_e{i}.pt"), map_location=device))
        model.eval()

        # get model parameters
        W, b = model.lin1.parameters()

        # get volume elements and top eigenvalue
        vol_elements, top_eigs = [], []
        for cur_scan in loader:
            cur_scan = cur_scan.to(device)
            vol_elements.append(determinant_analytic(cur_scan, W, b, args.nl).detach().cpu())
            top_eigs.append(top_eig_analytic(cur_scan, W, b, args.nl).detach().cpu())
        
        vol_elements = torch.hstack(vol_elements)
        top_eigs = torch.hstack(top_eigs)

        # save 
        torch.save(vol_elements, os.path.join(result_path, f'vol_elements_e{i}.pt'))
        torch.save(top_eigs, os.path.join(result_path, f"top_eigs_e{i}.pt"))

if __name__ == '__main__':
    main()
