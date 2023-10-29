"""
functions to process the name for logging models
"""

# load packages
from typing import List

def get_logging_name(args, mode: str) -> List[str]:
    """
    assign model logging name based on training configurations

    :param mode: indication of which files
        - linear_small: no batch size
        - linear_large: with batch size
        - deep: ResNet, SimCLR
    :return a tuple of strings, the base model name and regularized model name
    """
    if mode == 'linear_small':
        log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}' + \
           f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam{args.lam}_reg{args.reg}' + \
           f'_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}'
        base_log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}' + \
            f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam1_regNone' + \
            f'_ssNone_e{args.epochs}_b600_seed{args.seed}'

        # multi-layer
        if isinstance(args.hidden_dim, list) and len(args.hidden_dim) > 1: 
            log_name += f'_ml{args.max_layer}'
        
    elif mode == 'linear_large':
        log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}' + \
           f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam{args.lam}_reg{args.reg}' + (f'_m{args.m}' if args.reg == 'vol' else '') + \
           f'_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}'
        base_log_name = f'{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}' + \
            f'_lr{args.lr}_wd{args.wd}_nl{args.nl}_lam1_regNone' + \
            f'_ssNone_e{args.epochs}_b600_seed{args.seed}'
        
    else:
        raise NotImplementedError()
    
    return log_name, base_log_name