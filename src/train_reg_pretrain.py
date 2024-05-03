"""
pretrain on ImageNet using ResNet50
some codes are adapted from https://github.com/facebookresearch/FixRes
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
import torch.optim  as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# load file
from model import (
    ResNet50,
    top_eig_ub_transfer_update,
    spectral_ub_transfer_update,
    init_model_right_singular_conv
)
from utils import load_config, get_logging_name
from data import RASampler

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='imagenet1k', type=str, help='data to perform training on')

# model
parser.add_argument('--model', default='50', type=str, help="resnet model number", choices=["18", "34", "50", "101", "152"])

# training
parser.add_argument('--batch-size', default=64, type=int, help='the batchsize for training at each GPU')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate for the last layer')
parser.add_argument("--nl", default='ReLU', type=str, help='the nonlinearity throughout')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=1e-4, type=float, help='the weight decay overall')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--alpha', default=0.1, type=float, help='strength of l2sp')
parser.add_argument('--beta', default=0.01, type=float, help='strength for l2-norm new head in transfer architecture')
parser.add_argument('--lam', default=1e-4, type=float, help='the multiplier / strength of regularization')
parser.add_argument("--max-layer", default=None, type=int, 
                    help='the number of layers to regularize in ResNet architecture; None to regularize all'
                    )
parser.add_argument('--reg', default='None', type=str, help='the type of regularization', nargs='+')
parser.add_argument("--epochs", default=120, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=0, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--reg-freq', default=1, type=int, help='the frequency of imposing regularizations')
parser.add_argument('--reg-freq-update', default=None, type=int, 
                    help='the frequency of imposing convolution singular value regularization per parameter update: None means reg every epoch only'
                    )
parser.add_argument("--schedule", default=False, action='store_true', help='true to turn on cosine annealing lr scheduling')
parser.add_argument("--tmax", default=200, type=int, help='the T-Max parameter in cosine annealing learning rate scheduling')

# # iterative singular 
parser.add_argument('--iterative', action='store_true', default=False, help='True to turn on iterative method')
parser.add_argument('--eps', default=1e-4, type=float, help='the tolerance for stopping the iterative method')
parser.add_argument('--max-update', default=1, type=int, help='the maximum update iteration during training')

# logging
parser.add_argument('--log-epoch', default=20, type=int, help='logging frequency')
parser.add_argument('--log-model', default=20, type=int, help='model logging frequency')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)

# parameter compatibility check
if args.reg_freq_update is not None:
    assert args.reg_freq == 1, f"reg_freq_update{args.reg_freq_update} & reg_freq{args.reg_freq} not compatible"

# set up summary writer
log_name, base_log_name = get_logging_name(args, 'pretrain')
if "None" in args.reg: log_name = base_log_name
model_path = os.path.join(paths['model_dir'], log_name)
os.makedirs(model_path, exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

# load data
def load_data():
    if args.data == 'imagenet1k':
        from data import imagenet1k
        train_set, test_set = imagenet1k(paths['data_dir'])
        num_classes = 1000
        h, w = 224, 224
    elif args.data == 'cifar10':
        # for testing/debugging purpose only
        from data import cifar10
        train_set, test_set = cifar10(paths['data_dir'])
        num_classes = 10
        h, w = 32, 32
    else:
        raise NotImplementedError(f"{args.data} is not available")
    return train_set, test_set, num_classes, (h, w)

def update_conv(model, v_init):
    """individually update convolution layers"""
    if "spectral" in args.reg:
        spectral_ub_transfer_update(
            model, 
            max_layer=4, 
            lam=args.lam, 
            iterative=args.iterative, 
            v_init=v_init, 
            max_update=args.max_update, 
            tol=args.eps
        )
    if "eig-ub" in args.reg:
        if args.max_layer is None:
            max_layer = 4
        else:
            max_layer = args.max_layer
        top_eig_ub_transfer_update(
            model,
            max_layer=max_layer,
            lam=args.lam,
            iterative=args.iterative, 
            v_init=v_init, 
            max_update=args.max_update,
            tol=args.eps
        )

# ================= main driver training functions =================
def train(rank: int):
    """
    train a model with/without regularization
    
    :param rank: the current rank for training
    """
    # get GPU index
    num_tasks = torch.cuda.device_count()
    device_id = rank % torch.cuda.device_count()

    # load data
    train_set, test_set, num_classes, (h, w) = load_data()

    # batch to dataloader
    train_sampler = RASampler(train_set, num_tasks, rank, len(train_set), args.batch_size, repetitions=3, len_factor=2.0, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    test_sampler = RASampler(test_set, num_tasks, rank, len(test_set), args.batch_size, repetitions=1, len_factor=1.0, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler)
    if rank == 0: print(f'{args.data} data loaded')

    # get model
    nl = getattr(nn, args.nl)()
    model = ResNet50(num_classes=num_classes, nl=nl).to(device_id)

    # initialize
    if "None" in args.reg:
        pbar = tqdm(range(args.epochs + 1))

    # load from checkpoints
    else:
        pbar = tqdm(range(args.burnin, args.epochs + 1))

        # load model
        try:
            model.load_state_dict(torch.load(os.path.join(paths['model_dir'], base_log_name, f'model_e{args.burnin}.pt'), map_location=device_id))
        except:
            warnings.warn(f"Not finding base model {base_log_name}, retraining ...")
            # relapse back to full training
            pbar = tqdm(range(args.epochs + 1))
    
    # wrap to DDP
    ddp_model = DDP(model, device_ids=[device_id])
    # linear_scaled_lr = 8.0 * args.lr * args.batch_size * num_tasks / 512.0
    linear_scaled_lr = args.lr

    # get optimizer 
    opt = getattr(optim, args.opt)(
        ddp_model.parameters(), 
        lr=linear_scaled_lr, 
        momentum=args.mom, 
        weight_decay=args.wd
    )

    # linear scheduling
    if args.schedule:
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=30)

    loss_fn = nn.CrossEntropyLoss()

    # init model singular values if using power iteration (only keep in main rank)
    if args.iterative and rank == 0:
        if 'spectral' in args.reg or 'eig-ub' in args.reg:
            print("initialize top right singular direction")
            if "eig-ub" in args.reg and args.max_layer is not None: 
                max_layer = args.max_layer 
            else:
                max_layer = 4
            
            # set up initial guess dump path
            dump_path = os.path.join(model_path, f"resnet50pt_{h}_{w}_right_v_init")
            os.makedirs(dump_path, exist_ok=True)
            v_init = init_model_right_singular_conv(
                model, tol=args.eps, h=h, w=w, max_layer=max_layer, 
                dump_path=dump_path
            )
    else:
        v_init = {}
    
    # wait till initialization is finished for all ranks
    dist.barrier()

    # =========================
    # ------- training --------
    # =========================
    for i in pbar: 
        ddp_model.train()

        # update scheduling rate:
        if args.schedule:
            scheduler.step(i)

        total_train_loss, total_train_acc = 0, 0
        num_samples = 0
        for param_update_count, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.cuda(device_id, non_blocking=True), y_train.cuda(device_id, non_blocking=True)
            opt.zero_grad()
            y_pred_logits = ddp_model(X_train)
            train_loss = loss_fn(y_pred_logits, y_train)

            # add hard stop if nan
            if torch.isnan(train_loss): raise ValueError("loss is nan")

            # record
            total_train_loss += train_loss * len(X_train)
            total_train_acc += y_pred_logits.argmax(dim=-1).eq(y_train).sum()

            # step
            train_loss.backward()
            opt.step()
            num_samples += len(X_train)

            # regularization on convolution layers
            if (args.reg_freq_update is not None) and (param_update_count % args.reg_freq_update == 0):
                # use root rank to update conv layers
                if rank == 0: update_conv(model, v_init)

                # synchronize parameters (blocking)
                for param in ddp_model.parameters(): dist.broadcast(param.data, src=0)
            
            if param_update_count >= 5005 * 512 / (args.batch_size * num_tasks):
                break

        # reduce statistics in each rank 
        train_statistics = torch.tensor([total_train_loss, total_train_acc], device=device_id)
        dist.reduce(train_statistics, dst=0) # blocking reduction to main rank
        train_loss = train_statistics[0] / (num_samples * num_tasks)
        train_acc = train_statistics[1] / (num_samples * num_tasks)

        if i % args.log_epoch == 0:
            # testing
            ddp_model.eval()
            with torch.no_grad():
                local_test_loss, local_test_num_correct = 0, 0
                num_samples = 0
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.cuda(device_id, non_blocking=True), y_test.cuda(device_id, non_blocking=True)
                    y_test_pred_logits = ddp_model(X_test)
                    test_loss = loss_fn(y_test_pred_logits, y_test)

                    local_test_loss += test_loss * len(X_test)
                    local_test_num_correct += y_test_pred_logits.argmax(dim=-1).eq(y_test).sum()
                    num_samples += len(X_test)

                
                # gather statistics from all ranks
                test_statistics = torch.tensor([local_test_loss, local_test_num_correct], device=device_id)
                dist.reduce(test_statistics, dst=0) # sum up across all ranks and gather in root (blocking)
                test_loss = test_statistics[0] / (num_samples * num_tasks)
                test_acc = test_statistics[1] / (num_samples * num_tasks)

            # logging in root rank 
            if rank == 0:
                writer.add_scalar('train/loss', train_loss, i)
                writer.add_scalar('train/acc', train_acc, i)
                writer.add_scalar('test/loss', test_loss, i)
                writer.add_scalar('test/acc', test_acc, i)
                writer.flush()

                # # print
                pbar.set_description(f"epoch {i}")
                pbar.set_postfix({'tr_loss': train_loss.item(), 'tr_acc': train_acc.item(), 'te_loss': test_loss.item(), 'te_acc': test_acc.item()})

        # log model weights in the root rank
        if i % args.log_model == 0 and rank == 0:
            torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, f'model_e{i}.pt'))


def main():
    # initialize group of communications
    dist.init_process_group("nccl") # for FASRC
    rank = dist.get_rank()
    if rank == 0: print(f"DDP initialized, using {torch.cuda.device_count()} ranks")

    # train model
    train(rank)

    # finalize communications
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
