"""
train multiclass classification with regularization
Here we will use dataloader for batch training

to reduce runtime, we first train the model without a regularization,
and then read the model at some timestamp to train with regularizers

regularization is done after each minibatch update
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
    SLP, weights_init,
    init_model_right_singular,
    cross_lipschitz_regulerizer, 
    # volume_element_regularizer_autograd, 
    top_eig_regularizer_autograd,
    top_eig_ub_regularizer_autograd,
    top_eig_ub_pure_regularizer_autograd,
    spectral_ub_regularizer_autograd
)
from utils import load_config, get_logging_name

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on')
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=[20], type=int, help='the number of hidden units', nargs='+')

# training
parser.add_argument('--batch-size', default=1024, type=int, help='the batchsize for training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument("--opt", default='SGD', type=str, help='the type of optimizer')
parser.add_argument("--wd", default=0, type=float, help='weight decay')
parser.add_argument('--mom', default=0.9, type=float, help='the momentum')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default='None', type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=None, type=float, 
    help='the proportion of top samples for regularization (regularize samples with top eigenvalues or vol element)')
parser.add_argument("--m", default=None, type=int, help='vol element specific: keep the top m singular values to regularize only')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')
parser.add_argument('--max-layer', default=None, type=int,
                    help='the number of max layers to pull information from. None means the entire feature map')
parser.add_argument('--reg-freq', default=5, type=int, help='the frequency of imposing regularizations')
parser.add_argument('--reg-freq-update', default=None, type=int,
                    help='the frequency of imposing regularization per parameter update: None means reg every epoch only'
                    )

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
parser.add_argument('--scanbatchsize', default=20, type=int, help='the number of samples to batch compute jacobian for')
args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modify widths
if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1: args.hidden_dim = args.hidden_dim[0]

# set up summary writer
log_name, base_log_name = get_logging_name(args, 'linear_large')
if args.reg == "None": log_name = base_log_name
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

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
# initialize gaussian weights
weights_init(model)

# get optimizer
opt = getattr(optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom)

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
    

    # init model singular values
    if args.iterative:
        if args.reg == 'spectral':
            v_init = init_model_right_singular(model.model, tol=args.tol)
        elif args.reg == 'eig-ub' or args.reg == 'eig-ub-pure':
            v_init = init_model_right_singular(
                model.feature_map if hasattr(model, 'feature_map') else model.feature_maps[-1],
                tol=args.tol
            )
        print("initialize top right singular direction")
    else:
        v_init = {}

    # training
    for i in pbar:
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for parameter_update_count, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            opt.zero_grad()
            y_pred_logits = model(X_train)
            train_loss = loss_fn(y_pred_logits, y_train)
            
            # record
            total_train_loss += train_loss * len(X_train)
            total_train_acc += y_pred_logits.argmax(dim=-1).eq(y_train).sum()

            # add regularization
            if i > args.burnin and i % args.reg_freq == 0 and (
                (args.reg_freq_update is not None
                and parameter_update_count % args.reg_freq_update == 0) # anything else
                or (args.reg_freq_update is None) # reg_freq_update=1
            ):
                reg_loss = 0
                if args.reg == 'cross-lip':
                    reg_loss = cross_lipschitz_regulerizer(
                        model, X_train, is_binary=False)
                elif args.reg == 'eig':
                    reg_loss = top_eig_regularizer_autograd(
                        X_train, model.feature_map,
                        sample_size=args.sample_size, scanbatchsize=args.scanbatchsize
                    )
                elif args.reg == 'eig-ub':
                    reg_loss = top_eig_ub_regularizer_autograd(
                        X_train, model.feature_map,
                        max_layer=args.max_layer,
                        iterative=args.iterative, v_init=v_init, tol=args.tol, max_update=args.max_update
                    )
                elif args.reg == 'eig-ub-pure':
                    reg_loss = top_eig_ub_pure_regularizer_autograd(
                        model.feature_map,
                        max_layer=args.max_layer,
                        iterative=args.iterative, v_init=v_init, tol=args.tol, max_update=args.max_update
                    )
                elif args.reg == 'spectral':
                    reg_loss = spectral_ub_regularizer_autograd(
                        model,
                        iterative=args.iterative, v_init=v_init, tol=args.tol, max_update=args.max_update
                    )
                
                # add to loss 
                train_loss += args.lam * reg_loss

            # step
            train_loss.backward()
            opt.step()
        
        train_loss = total_train_loss / len(train_set)
        train_acc = total_train_acc / len(train_set)

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
