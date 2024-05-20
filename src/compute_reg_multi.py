"""
get key geometric quantities
"""

# load packages
import os
import argparse
import warnings
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# load file
from data import CustomScan, random_samples_by_targets
from model import SLP
from utils import load_config, get_logging_name, determinant_and_eig_autograd

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument(
    "--data", default="sin", type=str, help="data to perform training on"
)
parser.add_argument("--step", default=20, type=int, help="the number of steps")
parser.add_argument(
    "--test-size", default=0.5, type=float, help="the proportion of dataset for testing"
)

# model
parser.add_argument(
    "--hidden-dim", default=[20], type=int, help="the number of hidden units", nargs="+"
)

# training
parser.add_argument(
    "--batch-size", default=1024, type=int, help="the batchsize for training"
)
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--nl", default="Sigmoid", type=str, help="the nonlinearity of first layer"
)
parser.add_argument("--opt", default="SGD", type=str, help="the type of optimizer")
parser.add_argument("--wd", default=0, type=float, help="weight decay")
parser.add_argument("--mom", default=0.9, type=float, help="the momentum")
parser.add_argument(
    "--lam", default=1, type=float, help="the multiplier / strength of regularization"
)
parser.add_argument(
    "--reg", default="None", type=str, help="the type of regularization"
)
parser.add_argument(
    "--sample-size",
    default=None,
    type=float,
    help="the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)",
)
parser.add_argument(
    "--m",
    default=None,
    type=int,
    help="vol element specific: keep the top m singular values to regularize only",
)
parser.add_argument(
    "--epochs", default=1200, type=int, help="the number of epochs for training"
)
parser.add_argument(
    "--burnin",
    default=600,
    type=int,
    help="the period before which no regularization is imposed",
)
parser.add_argument(
    "--max-layer",
    default=None,
    type=int,
    help="the number of max layers to pull information from. None means the entire feature map",
)
parser.add_argument(
    "--reg-freq", default=5, type=int, help="the frequency of imposing regularizations"
)
parser.add_argument(
    "--reg-freq-update",
    default=None,
    type=int,
    help="the frequency of imposing regularization per parameter update: None means reg every epoch only",
)

# iterative singular
parser.add_argument(
    "--iterative",
    action="store_true",
    default=False,
    help="True to turn on iterative method",
)
parser.add_argument(
    "--tol",
    default=1e-6,
    type=float,
    help="the tolerance for stopping the iterative method",
)
parser.add_argument(
    "--max-update",
    default=10,
    type=int,
    help="the maximum update iteration during training",
)

# logging
parser.add_argument("--log-epoch", default=100, type=int, help="logging frequency")
parser.add_argument("--log-model", default=10, type=int, help="model logging frequency")

# technical
parser.add_argument("--tag", default="exp", type=str, help="the tag of batch of exp")
parser.add_argument("--seed", default=401, type=int, help="the random init seed")
parser.add_argument(
    "--scanbatchsize",
    default=20,
    type=int,
    help="the number of samples to batch compute jacobian for",
)

# plotting the sample points
parser.add_argument(
    "--sample-step",
    default=40,
    type=int,
    help="the number of samples along each axis for computing",
)
parser.add_argument(
    "--scan-batchsize",
    default=16,
    type=int,
    help="the number of samples to batch for computations",
)
parser.add_argument("--upper", default=1.0, type=float, help="the upper bound of vis")
parser.add_argument("--lower", default=-1.0, type=float, help="the lower bound of vis")
parser.add_argument(
    "--target-digits",
    default=[7, 6, 1],
    type=int,
    nargs="+",
    help="the target digits to generate plane from",
)
parser.add_argument("--vis-epochs", nargs='+', type=int, default=[200], help='the vis epochs')

args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modify widths
if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1:
    args.hidden_dim = args.hidden_dim[0]

# set up summary writer
# set up summary writer
log_name, base_log_name = get_logging_name(args, "linear_large")
if args.reg == "None":
    log_name = base_log_name
result_path = os.path.join(paths["result_dir"], log_name)


# load sample data
def load_sample_points():
    """generate a linear interpolation from 3 sample points"""
    if args.data == "mnist":
        from data import mnist

        train_set, test_set = mnist(paths["data_dir"], flatten=True)
    elif args.data == "fashion_mnist":
        from data import fashion_mnist

        train_set, test_set = fashion_mnist(paths["data_dir"], flatten=True)
    else:
        raise NotImplementedError(f"{args.data} not available")

    # get scan range
    # * randomly sample target digit from test set
    assert (
        len(args.target_digits) == 3
    ), "target digits does not have exactly two numbers"
    first_num_sample, second_num_sample, third_num_sample = random_samples_by_targets(
        test_set, targets=args.target_digits, seed=args.seed
    )
    # setup scan
    origin = (first_num_sample + second_num_sample + third_num_sample) / 3
    right_vec = second_num_sample - first_num_sample
    up_vec = ((first_num_sample + second_num_sample) / 2 - origin) / (
        1 / 2 / 3 ** (1 / 2)
    )

    l = torch.linspace(args.lower, args.upper, steps=args.sample_step)
    y = torch.linspace(args.lower, args.upper, steps=args.sample_step)

    raw = torch.cartesian_prod(l, y)
    scan = (
        origin + raw[:, [0]] * right_vec + raw[:, [1]] * up_vec
    )  # orthogonal decomposition

    # save mid and endpoints
    torch.save(first_num_sample, os.path.join(result_path, "point_one.pt"))
    torch.save(second_num_sample, os.path.join(result_path, "point_two.pt"))
    torch.save(third_num_sample, os.path.join(result_path, "point_three.pt"))
    torch.save(origin, os.path.join(result_path, "origin.pt"))

    # add 3 sides
    t = torch.arange(args.sample_step, device=first_num_sample.device).reshape(-1, 1)
    slice_12 = (
        second_num_sample - first_num_sample
    ) * t / args.sample_step + first_num_sample
    slice_13 = (
        third_num_sample - first_num_sample
    ) * t / args.sample_step + first_num_sample
    slice_23 = (
        third_num_sample - second_num_sample
    ) * t / args.sample_step + second_num_sample

    # append to scan
    scan = torch.vstack((scan, slice_12, slice_13, slice_23))

    # pack into a dataloader
    dataset = CustomScan(scan)
    return dataset


# get model
def load_model() -> nn.Module:
    nl = getattr(nn, args.nl)()
    model = SLP(input_dim=784, width=args.hidden_dim, output_dim=10, nl=nl).to(device)
    return model


def main():
    # get samples
    scan_dataset = load_sample_points()
    loader = DataLoader(
        scan_dataset, args.scan_batchsize, shuffle=False, drop_last=False
    )

    # get model
    model = load_model()
    pbar = tqdm(args.vis_epochs)

    # training
    for i in pbar:
        # load model parameters
        model.load_state_dict(
            torch.load(
                os.path.join(paths["model_dir"], log_name, f"model_e{i}.pt"),
                map_location=device,
            ),
        )
        model.eval()

        # get feature map
        feature_map = model.feature_map

        vol_elements, top_eigs = [], []
        for cur_scan in loader:
            cur_scan = cur_scan.to(device)
            cur_eigs, cur_vol = determinant_and_eig_autograd(cur_scan, feature_map)
            vol_elements.append(cur_vol.detach().cpu())
            top_eigs.append(cur_eigs.detach().cpu())

        vol_elements = torch.hstack(vol_elements)
        top_eigs = torch.hstack(top_eigs)

        # save
        torch.save(vol_elements, os.path.join(result_path, f"vol_elements_e{i}.pt"))
        torch.save(top_eigs, os.path.join(result_path, f"top_eigs_e{i}.pt"))


if __name__ == "__main__":
    main()
