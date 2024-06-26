"""
train network of different regularization, using 2D dataset for binary classification

note that for binary classification, we still use CrossEntropyLoss (i.e. output dim 2)
for consistency in comparing across different regularizations

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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# load file
from data import CustomDataset
from model import (
    SLP, MLP, weights_init,
    init_model_right_singular,
    cross_lipschitz_regulerizer, 
    volume_element_regularizer, 
    top_eig_regularizer,
    top_eig_regularizer_autograd,
    top_eig_ub_regularizer_autograd,
    spectral_ub_regularizer_autograd
)
from utils import load_config, get_logging_name

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
args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modify widths and initialize model
nl = getattr(nn, args.nl)()
if isinstance(args.hidden_dim, list) and len(args.hidden_dim) == 1: 
    # single-hidden-layer
    args.hidden_dim = args.hidden_dim[0]
    model = SLP(input_dim=2, width=args.hidden_dim, output_dim=2, nl=nl).to(device)
else:
    # multi-hidden-layer
    model = MLP([2, *args.hidden_dim, 2], nl=nl).to(device)

# set up summary writer
log_name, base_log_name = get_logging_name(args, 'linear_small')
if args.reg == 'None': log_name = base_log_name
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

# load data
def load_data():
    if args.data == 'linear':
        from data import load_linear_boundary
        X_train, X_test, y_train, y_test = load_linear_boundary(args.step, args.test_size, args.seed)
    elif args.data == 'xor':
        from data import load_xor_boundary
        X_train, X_test, y_train, y_test = load_xor_boundary(args.step, args.test_size, args.seed)
    elif args.data == 'xor_symmetric':
        from data import load_xor_symmetric
        X_train, X_test, y_train, y_test = load_xor_symmetric()
    elif args.data == 'xor_noisy':
        from data import load_xor_noisy
        X_train, X_test, y_train, y_test = load_xor_noisy(args.step, 0.2, args.seed) # * tune variation here
    elif args.data == 'sin':
        from data import load_sin_boundary
        X_train, X_test, y_train, y_test = load_sin_boundary(args.step, args.test_size, args.seed)
    elif args.data == 'sin-random':
        from data import load_sin_random
        X_train, X_test, y_train, y_test = load_sin_random(args.step, args.test_size, args.seed)
    else:
        raise NotImplementedError(f'{args.data} not available')

    # convert to long for CrossEntropyLoss
    y_train, y_test = y_train.long(), y_test.long()
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
print(f'{args.data} data loaded')

# convert to data loader
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)


# get optimizer
weights_init(model)
opt = getattr(optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom)


def train():
    """full batch training"""
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
                map_location=device)
            )
        except:
            warnings.warn('Not finding base model, retraining ...')
            pbar = tqdm(range(args.epochs + 1)) # relapse back to full training
        

    # init model singular values
    if args.iterative:
        if args.reg == 'spectral':
            v_init = init_model_right_singular(model.model, tol=args.tol)
        elif args.reg == 'eig-ub':
            v_init = init_model_right_singular(
                model.feature_map if hasattr(
                    model, 'feature_map') else model.feature_maps[-1],
                tol=args.tol
            )
        print("initialize top right singular direction")
    else:
        v_init = {}
    
    # training
    for i in pbar:
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for batch_X_train, batch_y_train in train_loader:
            opt.zero_grad()
            y_pred_logits = model(batch_X_train)
            train_loss = loss_fn(y_pred_logits, batch_y_train)
            train_acc = (y_pred_logits.argmax(dim=-1) == batch_y_train).to(torch.float32).mean()

            # record 
            total_train_loss += train_loss * len(batch_X_train)
            total_train_acc += y_pred_logits.argmax(dim=-1).eq(batch_y_train).sum()

            # add regularization
            if i > args.burnin:
                reg_loss = 0
                if args.reg == 'cross-lip':
                    reg_loss = cross_lipschitz_regulerizer(
                        model, batch_X_train, is_binary=True, sample_size=args.sample_size)
                elif args.reg == 'vol':
                    reg_loss = volume_element_regularizer(
                        model, batch_X_train, sample_size=args.sample_size)
                elif args.reg == 'eig-analytic':
                    reg_loss = top_eig_regularizer(model, batch_X_train, sample_size=args.sample_size)
                elif args.reg == 'eig':
                    feature_map = model.feature_map if hasattr(model, 'feature_map') else model.feature_maps[-1]
                    reg_loss = top_eig_regularizer_autograd(
                        batch_X_train, feature_map, sample_size=args.sample_size, max_layer=args.max_layer
                    )
                elif args.reg == 'eig-ub':
                    feature_map = model.feature_map if hasattr(model, 'feature_map') else model.feature_maps[-1]
                    reg_loss = top_eig_ub_regularizer_autograd(
                        batch_X_train, feature_map, max_layer=args.max_layer,
                        iterative=args.iterative, v_init=v_init, tol=args.tol, max_update=args.max_update
                    )
                elif args.reg == 'spectral':
                    reg_loss = spectral_ub_regularizer_autograd(
                        model,
                        iterative=args.iterative, v_init=v_init, tol=args.tol, max_update=args.max_update
                    )
                # add to train loss
                train_loss += reg_loss * args.lam

            # update 
            train_loss.backward()
            opt.step()
        
        train_loss = total_train_loss / len(X_train)
        train_acc = total_train_acc / len(X_train)


        if i % args.log_epoch == 0:
            # testing 
            model.eval()
            with torch.no_grad():
                y_test_pred_logits = model(X_test)
                test_loss = loss_fn(y_test_pred_logits, y_test)
                test_acc = y_test_pred_logits.argmax(dim=-1).eq(y_test).to(torch.float32).mean()
        
            # logging
            writer.add_scalar('train/loss', train_loss, i)  # only the last batch 
            writer.add_scalar('train/acc', train_acc, i)    # only the last batch
            writer.add_scalar('test/loss', test_loss, i)
            writer.add_scalar('test/acc', test_acc, i)
            writer.flush()

            # # print
            # print(f'-- epoch {i}: train_loss: {train_loss.item():.4f}, train_acc: {train_acc.item():.4f}')
            # print(f'-- epoch {i}: test_loss:  {test_loss.item():.4f}, test_acc:  {test_acc.item():.4f}\n')
            pbar.set_description(f"epoch {i}")
            pbar.set_postfix(
                {'tr_loss': train_loss.item(), 'tr_acc': train_acc.item(), 'te_loss': test_loss.item(), 'te_acc': test_acc.item()})

        # save model
        if i % args.log_model == 0:
            torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, f'model_e{i}.pt'))


def main():
    train()

if __name__ == '__main__':
    main()
