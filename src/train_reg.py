"""
train network of different regularization, using 2D dataset for binary classification
"""

# load packages
import os 
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# load file
from model import SLP, cross_lipschitz_regulerizer, volume_element_regularizer
from utils import load_config

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='sin', type=str, help='data to perform training on', choices=['linear', 'xor', 'sin'])
parser.add_argument("--step", default=20, type=int, help='the number of steps')
parser.add_argument("--test-size", default=0.5, type=float, help='the proportion of dataset for testing')

# model
parser.add_argument('--hidden-dim', default=20, type=int, help='the number of hidden units')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--nl", default='Sigmoid', type=str, help='the nonlinearity of first layer')
parser.add_argument('--lam', default=1, type=float, help='the multiplier / strength of regularization')
parser.add_argument('--reg', default=None, type=str, help='the type of regularization')
parser.add_argument('--sample-size', default=50, type=int, help='the number of samples for vol element reg')
parser.add_argument("--epochs", default=1200, type=int, help='the number of epochs for training')
parser.add_argument('--burnin', default=600, type=int, help='the period before which no regularization is imposed')

# logging
parser.add_argument('--log-epoch', default=100, type=int, help='logging frequency')

# technical
parser.add_argument('--tag', default='exp', type=str, help='the tag of batch of exp')
parser.add_argument('--seed', default=401, type=int, help='the random init seed')
args = parser.parse_args()

torch.manual_seed(args.seed)

# read paths
paths = load_config(args.tag)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up summary writer
log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_lr{args.lr}_nl{args.nl}_lam{args.lam}_reg{args.reg}_e{args.epochs}_b{args.burnin}_seed{args.seed}'
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
    elif args.data == 'sin':
        from data import load_sin_boundary
        X_train, X_test, y_train, y_test = load_sin_boundary(args.step, args.test_size, args.seed)
    else:
        raise NotImplementedError(f'{args.data} not available')
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)
print(f'{args.data} data loaded')

# get model
nl = getattr(nn, args.nl)()
model = SLP(width=args.hidden_dim, nl=nl).to(device)

# get optimizer
opt = SGD(model.parameters(), lr=args.lr)

def train():
    """
    full batch training
    """
    loss_fn = nn.BCELoss()
    sig = nn.Sigmoid()
    model.train()
    for i in range(args.epochs + 1):
        opt.zero_grad()
        y_pred_logits = sig(model(X_train)).flatten()
        train_loss = loss_fn(y_pred_logits, y_train)
        train_acc = ((y_pred_logits >= 0.5).to(int) == y_train).to(torch.float32).mean()

        if i % args.log_epoch == 0:
            # testing 
            with torch.no_grad():
                y_test_pred_logits = sig(model(X_test)).flatten()
                test_loss = loss_fn(y_test_pred_logits, y_test)
                test_acc = ((y_test_pred_logits >= 0.5).to(int) == y_test).to(torch.float32).mean()
        
            # logging
            writer.add_scalar('train/loss', train_loss, i)
            writer.add_scalar('train/acc', train_acc, i)
            writer.add_scalar('test/loss', test_loss, i)
            writer.add_scalar('test/acc', test_acc, i)
            writer.flush()

            # print 
            print(f'-- epoch {i}: train_loss: {train_loss.item():.4f}, train_acc: {train_acc.item():.4f}')
            print(f'-- epoch {i}: test_loss:  {test_loss.item():.4f}, test_acc:  {test_acc.item():.4f}\n')

        # regularization
        if i > args.burnin:
            if args.reg == 'cross-lip':
                reg_loss = cross_lipschitz_regulerizer(model, X_train, is_binary=True)
            elif args.reg == 'vol':
                reg_loss = volume_element_regularizer(model, X_train)
            elif args.reg == 'vol-sample':
                X_train_samples = X_train[torch.randperm(len(X_train))[:int(args.sample_size * len(X_train))]]
                reg_loss = volume_element_regularizer(model, X_train_samples)
            elif args.reg == 'weight': # equivalent to weight decay
                reg_loss = list(model.lin1.parameters())[0].norm(dim=1).square().mean()
            elif args.reg == 'vol-weight':
                reg_loss = list(model.lin1.parameters())[0].norm(dim=1).square().mean() * \
                    volume_element_regularizer(model, X_train)
            elif args.reg == 'vol-sample-weight':
                X_train_samples = X_train[torch.randperm(len(X_train))[:int(args.sample_size * len(X_train))]]
                reg_loss = list(model.lin1.parameters())[0].norm(dim=1).square().mean() * \
                    volume_element_regularizer(model, X_train_samples)
        else:
            reg_loss = 0
        
        # step
        train_loss += reg_loss * args.lam
        train_loss.backward()
        opt.step()

        # TODO: log other stuffs such as sample gradient and/or hessian
    # TODO: log model at some stages
    # log final model
    torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, 'model.pt'))
    print('final model saved!')


def main():
    train()

if __name__ == '__main__':
    main()
