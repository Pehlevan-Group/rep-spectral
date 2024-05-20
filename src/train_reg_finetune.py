"""
resnet50 transfer learning task, trained with different regularizations
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
    ResNet50Pretrained,
    l2sp_transfer,
    bss_transfer,
    # top_eig_ub_transfer,
    # spectral_ub_transfer
    top_eig_ub_transfer_update,
    spectral_ub_transfer_update,
    init_model_right_singular_conv
)
from utils import load_config, get_logging_name

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
parser.add_argument("--wd", default=0., type=float, help='the weight decay overall')
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
parser.add_argument("--schedule", default=False, action='store_true', help='true to turn on cosine annealing lr scheduling')
parser.add_argument("--tmax", default=200, type=int, help='the T-Max parameter in cosine annealing learning rate scheduling')

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

# using custom pretrained model 
parser.add_argument('--custom-pretrain', default=False, action='store_true', help='true to turn on reading from custom generated weights')
parser.add_argument("--pretrain-wt-path", type=str, help='the pretrain weight paths to load model from')

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
if "None" in args.reg: log_name = base_log_name
model_path = os.path.join(paths['model_dir'], log_name)
os.makedirs(model_path, exist_ok=True)
writer = SummaryWriter(os.path.join(paths['result_dir'], log_name))

# load data
def load_data():
    if args.data == 'dog':
        from data import load_dog
        train_set, test_set = load_dog(paths['data_dir'])
        num_classes = 120
        h = w = 224
    elif args.data == 'flower':
        from data import load_flower 
        train_set, test_set = load_flower(paths['data_dir'])
        num_classes = 102
        h = w = 224
    elif args.data == 'indoor':
        from data import load_indoor 
        train_set, test_set = load_indoor(paths['data_dir'])
        num_classes = 67
        h = w = 224
    elif args.data == 'cifar10':
        from data import cifar10
        train_set, test_set = cifar10(paths['data_dir'])
        num_classes = 10
        h = w = 32
    elif args.data == 'cifar10_resized':
        from data import cifar10_resized
        train_set, test_set = cifar10_resized(paths['data_dir'])
        num_classes = 10
        h = w = 224
    else:
        raise NotImplementedError(f"{args.data} is not available")
    return train_set, test_set, num_classes, (h, w)

train_set, test_set, num_classes, (h, w) = load_data()

# batch to dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size * 4, shuffle=False)
print(f'{args.data} data loaded')

# get model
if args.custom_pretrain:
    from model import ResNet50
    model = ResNet50(1000, small_input=(args.data == "cifar10")).to(device)
    backbone_parameters = list(model.parameters())[:-2]
    fc_parameters = list(model.parameters())[-2:]
    # TODO: check if registered properly
    print("initialize with ResNet50")
else:
    model = ResNet50Pretrained(
        num_classes, 
        small_conv1=args.data=='cifar10' # change 'conv1' layer for small dimensional data
    ).to(device)
    backbone_parameters = model.model.parameters()
    fc_parameters = model.fc.parameters()

# initialize
if args.custom_pretrain:
    pbar = tqdm(range(args.epochs + 1))

    # load from custom pretrained model
    # ! hard code path, change to regularized here
    model.load_state_dict(
        # torch.load("/n/pehlevan_lab/Users/shengy/geometry/GeomNet/model/pretrain/imagenet1k_m50_bs128_lr0.02_optSGD_wd0.0001_mom0.9_nlReLU_regNone_e120_seed0/model_e120.pt",
        # torch.load("/n/pehlevan_lab/Users/shengy/geometry/GeomNet/model/pretrain/imagenet1k_m50_bs128_lr0.02_optSGD_wd0.0001_mom0.9_nlReLU_lam0.001_reg['eig-ub']_e120_b80_seed0_rf1_ru80/model_e120.pt",
        torch.load(args.pretrain_wt_path,
        map_location=device),
    )
    # change output dimension
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True, device=device)

    print("loading from custom pretrained model")
else:
    if "None" in args.reg:
        pbar = tqdm(range(args.epochs + 1))
    else:
        # start training from burnin (empty pass for learning rates scheduling)
        pbar = tqdm(range(args.epochs + 1))

        # load model
        try:
            model.load_state_dict(
                torch.load(os.path.join(paths['model_dir'], base_log_name, f'model_e{args.burnin}.pt'),
                            map_location=device),
            )
        except:
            warnings.warn(f"Not finding base model {base_log_name}, retraining ...")
            # relapse back to full training
            pbar = tqdm(range(args.epochs + 1))

# get optimizer 
# backbone parameters
opt_backbone = getattr(optim, args.opt)(
    backbone_parameters,
    lr=args.lr / 5,  # ! much smaller learning rate for the pretraiend layers
    weight_decay=args.wd, 
    momentum=args.mom
)
# fully connected linear head
opt_fc = getattr(optim, args.opt)(
    fc_parameters, 
    lr=args.lr, 
    momentum=args.mom, 
    weight_decay=args.wd
)
if args.schedule:
    scheduler_backbone = optim.lr_scheduler.CosineAnnealingLR(opt_backbone, T_max=args.tmax)
    scheduler_fc = optim.lr_scheduler.CosineAnnealingLR(opt_fc, T_max=args.tmax)

loss_fn = nn.CrossEntropyLoss()

# init model singular values if using power iteration
if args.iterative:
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

# =================== regularization updates =====================
def get_reg_loss(model: ResNet50Pretrained, features: torch.Tensor) -> torch.Tensor: 
    """
    compute regularization loss
    in transfer learning, the regularizations are additive
    """
    reg_loss = 0
    if "l2sp" in args.reg: 
        reg_loss += l2sp_transfer(model, args.alpha, args.beta)
    if "bss" in args.reg:
        reg_loss += args.lam * bss_transfer(features)
    
    if reg_loss == 0: 
        return None 
    else:
        return reg_loss
    
def update_conv(model: ResNet50Pretrained):
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
def train():
    """train a model with/without regularization"""

    # training
    for i in pbar:
        # update scheduling rate
        if args.schedule:
            scheduler_backbone.step(i)
            scheduler_fc.step(i)
        
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for param_update_count, (X_train, y_train) in enumerate(train_loader):
            X_train, y_train = X_train.to(device), y_train.to(device)
            opt_backbone.zero_grad()
            opt_fc.zero_grad()
            if args.custom_pretrain:
                features, y_pred_logits = model.forward(X_train, return_features=True)
            else:
                features, y_pred_logits = model(X_train)
            train_loss = loss_fn(y_pred_logits, y_train)

            # add hard stop if nan
            if torch.isnan(train_loss): raise ValueError("loss is nan")

            # record
            total_train_loss += train_loss * len(X_train)
            total_train_acc += y_pred_logits.argmax(dim=-1).eq(y_train).sum()

            # regularization on predecessors (every parameter update)
            # if (args.reg_freq_update is not None) and (param_update_count % args.reg_freq_update == 0):
            reg_loss = get_reg_loss(model, features)
            if reg_loss is not None:
                train_loss += reg_loss

            # step
            train_loss.backward()
            opt_backbone.step()
            opt_fc.step()

            # regularization on convolution layers
            if (args.reg_freq_update is not None) and (param_update_count % args.reg_freq_update == 0):
                update_conv(model)

        train_loss = total_train_loss / len(train_set)
        train_acc = total_train_acc / len(train_set)

        # lr scheduling
        if args.schedule:
            scheduler_backbone.step()
            scheduler_fc.step()

        # regularization after each epoch, if not updated on a per parameter update basis
        if (i > args.burnin) and (args.reg_freq_update is None) and (i % args.reg_freq == 0):
            raise NotImplementedError("updating on per epoch basis is not supported, since features gradient will be discarded")
            # regularization_update(model, features, opt_backbone, opt_fc) # * only last batch of features is used

        if i % args.log_epoch == 0:
            # testing
            model.eval()
            with torch.no_grad():
                total_test_loss, total_test_acc = 0, 0
                for X_test, y_test in test_loader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    if args.custom_pretrain:
                        y_test_pred_logits = model.forward(X_test)
                    else:
                        _, y_test_pred_logits = model(X_test)
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
            pbar.set_postfix({'tr_loss': train_loss.item(), 'tr_acc': train_acc.item(), 'te_loss': test_loss.item(), 'te_acc': test_acc.item()})

        if i % args.log_model == 0:
            # log final model
            torch.save(model.state_dict(), os.path.join(paths['model_dir'], log_name, f'model_e{i}.pt'))


def main():
    train()


if __name__ == '__main__':
    main()
