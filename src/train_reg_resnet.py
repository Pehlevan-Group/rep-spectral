"""
train regularized ResNet model

in ResNet model, getting the singular values of convolution layer is expensive.
We no longer regularize on a per parameter update basis but regularize every $k$
number of parameter updates, where $k$ is a tunable parameter under `args.reg_freq_update`
Alternatively, we can regularize only after every epoch as well, using `args.reg_freq`
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
from torch.utils.tensorboard import SummaryWriter

# load file
from model import (
    top_eig_ub_regularizer_conv,
    spectral_ub_regularizer_conv,
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
)
from utils import load_config, get_logging_name

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='cifar10', type=str, help='data to perform training on')

# model
parser.add_argument('--model', default='34', type=str, help="resnet model number", choices=["18", "34", "50", "101", "152"],)

# training
parser.add_argument('--batch-size', default=1024, type=int, help='the batchsize for training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument("--nl", default='GELU', type=str, help='the nonlinearity throughout')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=0, type=float, help='weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--lam', default=1e-4, type=float, help='the multiplier / strength of regularization')
parser.add_argument("--max-layer", default=None, type=int, 
                    help='the number of layers to regularize in ResNet architecture; None to regularize all'
                    )
parser.add_argument('--reg', default='None', type=str, help='the type of regularization')
parser.add_argument("--epochs", default=200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=180, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--reg-freq', default=1, type=int, help='the frequency of imposing regularizations')
parser.add_argument('--reg-freq-update', default=None, type=int, 
                    help='the frequency of imposing regularization per parameter update: None means reg every epoch only'
                    )

# # iterative singular 
parser.add_argument('--iterative', action='store_true', default=False, help='True to turn on iterative method')
# parser.add_argument('--tol', default=1e-6, type=float, help='the tolerance for stopping the iterative method')
# parser.add_argument('--max-update', default=10, type=int, help='the maximum update iteration during training')

# logging
parser.add_argument('--log-epoch', default=5, type=int, help='logging frequency')
parser.add_argument('--log-model', default=20, type=int, help='model logging frequency')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter compatibility check
if args.reg_freq_update is not None: 
    assert args.reg_freq == 1, f"reg_freq_update{args.reg_freq_update} & reg_freq{args.reg_freq} not compatible"

# set up summary writer
log_name, base_log_name = get_logging_name(args, 'conv')
if args.reg == "None": log_name = base_log_name
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

# load data 
def load_data():
    if args.data == 'cifar10':
        from data import cifar10
        train_set, test_set = cifar10(paths['data_dir'])
    else:
        raise NotImplementedError(f"{args.data} is not available")
    return train_set, test_set

train_set, test_set = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
print(f'{args.data} data loaded')

# get model
nl = getattr(nn, args.nl)()
model = eval(f"ResNet{args.model}")(nl=nl).to(device)

# get optimizer
opt = getattr(optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom)

def get_reg_loss(model: nn.Module) -> torch.Tensor:
    """
    compute regularization loss
    
    :return None if no regularization imposed
    """
    reg_loss = None
    if args.reg == 'eig-ub':
        # parse max_layer argument
        if args.max_layer is None: 
            max_layer = 4 # total number of layers in any ResNet architectures
        else:
            max_layer = args.max_layer
        reg_loss = top_eig_ub_regularizer_conv(model, max_layer=max_layer)
    elif args.reg == 'spectral':
        reg_loss = spectral_ub_regularizer_conv(model)
    return reg_loss

def train():
    """train a model with/without regularization"""
    loss_fn = nn.CrossEntropyLoss()
    
    # initialize 
    if args.reg == 'None':
        pbar = tqdm(range(args.epochs + 1)) 
    else:
        # start training from burnin 
        pbar = tqdm(range(args.burnin, args.epochs + 1))
        
        # load model
        try: 
            model.load_state_dict(
                torch.load(os.path.join(paths['model_dir'], base_log_name, f'model_e{args.burnin}.pt'),
                map_location=device),
            )
        except:
            warnings.warn('Not finding base model, retraining ...')
            pbar = tqdm(range(args.epochs + 1)) # relapse back to full training
    

    # TODO: iterative method? 

    # training
    for i in pbar:
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for param_update_count, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            opt.zero_grad()
            y_pred_logits = model(X_train)
            train_loss = loss_fn(y_pred_logits, y_train)
            
            # record
            total_train_loss += train_loss * len(X_train)
            total_train_acc += y_pred_logits.argmax(dim=-1).eq(y_train).sum()

            # regularization on a per parameter update basis
            if (args.reg_freq_update is not None) and (param_update_count % args.reg_freq_update == 0):
                reg_loss = get_reg_loss(model)
                if reg_loss is not None:
                    train_loss += args.lam * reg_loss

            # step
            train_loss.backward()
            opt.step()
        
        train_loss = total_train_loss / len(train_set)
        train_acc = total_train_acc / len(train_set)

        # regularization after each epoch, if not updated on a per parameter update basis
        if (i > args.burnin) and (args.reg_freq_update is not None) and (i % args.reg_freq == 0):
            reg_loss = get_reg_loss(model)

            if reg_loss is not None:
                opt.zero_grad()
                reg_loss *= args.lam 
                reg_loss.backward()
                opt.step()

        if i % args.log_epoch == 0:
            # testing 
            model.eval()
            with torch.no_grad():
                total_test_loss, total_test_acc = 0, 0
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    y_test_pred_logits = model(X_test)
                    test_loss = loss_fn(y_test_pred_logits, y_test)

                    total_test_loss += test_loss * len(X_test)
                    total_test_acc += y_test_pred_logits.argmax(dim=-1).eq(y_test).sum()
                test_loss = total_test_loss / len(test_set)
                test_acc = total_test_acc / len(test_set)
        
            # logging
            writer.add_scalar('train/loss', train_loss, i)
            writer.add_scalar('train/acc', train_acc, i)
            writer.add_scalar('test/loss', test_loss, i)
            writer.add_scalar('test/acc', test_acc, i)
            writer.flush()

            # # print 
            # print(f'-- epoch {i}: train_loss: {train_loss.item():.4f}, train_acc: {train_acc.item():.4f}')
            # print(f'-- epoch {i}: test_loss:  {test_loss.item():.4f}, test_acc:  {test_acc.item():.4f}\n')
            pbar.set_description(f"epoch {i}")
            pbar.set_postfix(
                {'tr_loss': train_loss.item(), 'tr_acc': train_acc.item(), 'te_loss': test_loss.item(), 'te_acc': test_acc.item()})

        if i % args.log_model == 0:
            # log final model
            torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, f'model_e{i}.pt'))


def main():
    train()

if __name__ == '__main__':
    main()
