"""
train SimCLR using resnet backbone
"""

# load packages
import os 
import argparse
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim  as optim
from torch.utils.tensorboard import SummaryWriter

# load file
from data import cifar10_contrastive
from model import (
    top_eig_ub_regularizer_conv,
    ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
    SimCLR,
    get_contrastive_acc
)
from utils import load_config, get_logging_name

# arguments
parser = argparse.ArgumentParser()
# data
parser.add_argument("--data", default='cifar10', type=str, help='data to perform training on')

# model
parser.add_argument('--model', default='34', type=str, help="resnet model number", choices=["18", "34", "50", "101", "152"])

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

# SimCLR specific parameters
parser.add_argument("--temperature", default=0.07, type=float, help="the temperature in softmax")

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
log_name, base_log_name = get_logging_name(args, 'simclr')
if args.reg == "None": log_name = base_log_name
model_path = os.makedirs(os.path.join(paths['model_dir'], log_name), exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

# load data
def load_data():
    if args.data == "cifar10":
        train_dataset, unaugmented_train_dataset, test_dataset = cifar10_contrastive(
            # SIMCLR style data augmentation
            paths["data_dir"], 
            transformation="SimClr"
        )
    else:
        raise NotImplementedError(f"dataset {args.data} not available")
    return train_dataset, unaugmented_train_dataset, test_dataset

train_set, unaugmented_train_dataset, test_set = load_data()

# batch to dataloader 
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
unaug_loader = DataLoader(unaugmented_train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
print(f"{args.data} data loaded")

# store train and test labels
train_labels, test_labels = [], []
for _, labels in unaug_loader:
    train_labels.append(labels)
for _, labels in test_loader:
    test_labels.append(labels)

train_labels = torch.concat(train_labels, dim=0)
test_labels = torch.concat(test_labels, dim=0)

# get model
nl = getattr(nn, args.nl)()
backbone = eval(f"ResNet{args.model}")(nl=nl).to(device)
model = SimCLR(backbone, args.batch_size, nl=nl).to(device)
nce_loss = model.nce_loss

# get optimizer
opt = getattr(optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mom)

def get_reg_loss(model: nn.Module) -> torch.Tensor:
    """
    compute regularization loss
    * for contrastive models, we have only one regularization method
    
    :return None if no regularization imposed
    """
    reg_loss = None
    if args.reg == 'eig-ub':
        # parse max_layer argument
        if args.max_layer is None:
            max_layer = 4  # total number of layers in any ResNet architectures
        else:
            max_layer = args.max_layer
        reg_loss = top_eig_ub_regularizer_conv(model, max_layer=max_layer)
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
        total_train_loss = 0
        total, correct = 0, 0
        for param_update_count, (inputs, _) in train_loader:
            # stack multiple augmentations
            inputs = torch.cat(inputs, dim=0).to(device)
            opt.zero_grad()

            projections = model(inputs)
            logits, labels = nce_loss(projections)
            logits = logits / args.temperature
            train_loss = loss_fn(logits, labels)

            with torch.no_grad():
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # record
            total_train_loss += train_loss

            # regularization on a per parameter update basis
            if (args.reg_freq_update is not None) and (param_update_count % args.reg_freq_update == 0):
                reg_loss = get_reg_loss(model)
                if reg_loss is not None:
                    train_loss += args.lam * reg_loss

            # step
            train_loss.backward()
            opt.step()
        
        train_loss = total_train_loss
        train_acc = correct / total 

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
                # get acc 
                downstream_train_acc, downstream_test_acc = get_contrastive_acc(
                    model, unaug_loader, train_labels, test_loader, test_labels,
                    device=device, random_state=args.seed
                )
        
            # logging
            writer.add_scalar('train/loss', train_loss, i)
            writer.add_scalar('train/acc', downstream_train_acc, i)
            # writer.add_scalar('test/loss', test_loss, i) # * very expensive to evaluate test loss
            writer.add_scalar('test/acc', downstream_test_acc, i)
            writer.flush()

            # # print 
            # print(f'-- epoch {i}: train_loss: {train_loss.item():.4f}, train_acc: {train_acc.item():.4f}')
            # print(f'-- epoch {i}: test_loss:  {test_loss.item():.4f}, test_acc:  {test_acc.item():.4f}\n')
            pbar.set_description(f"epoch {i}")
            pbar.set_postfix(
                {'tr_loss': train_loss.item(), "tr_acc": train_acc, "tr_dacc": downstream_train_acc, 'te_dacc': downstream_test_acc})

        if i % args.log_model == 0:
            # log final model
            torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, f'model_e{i}.pt'))


def main():
    train()

if __name__ == '__main__':
    main()
