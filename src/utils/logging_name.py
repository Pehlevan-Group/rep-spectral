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
        - conv: ResNet, SimCLR
    :return a tuple of strings, the base model name and regularized model name
    """
    # modify reg name
    reg_name = args.reg
    if reg_name in ["spectral", "eig-ub", "eig-ub-pure"] and args.iterative:
        reg_name += "-iterative"
    elif reg_name == "vol":
        reg_name += f"_m{args.m}"

    # choose log and unregularized log name
    if mode == "linear_small":
        log_name = (
            f"{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}"
            + f"_lr{args.lr}_opt{args.opt}_wd{args.wd}_mom{args.mom}_nl{args.nl}_lam{args.lam}"
            + f"_reg{reg_name}"
            + f"_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}"
        )
        base_log_name = (
            f"{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}"
            + f"_lr{args.lr}_opt{args.opt}_wd{args.wd}_mom{args.mom}_nl{args.nl}_regNone"
            + f"_e{args.epochs}_seed{args.seed}"
        )

        # multi-layer
        if isinstance(args.hidden_dim, list) and len(args.hidden_dim) > 1:
            log_name += f"_ml{args.max_layer}"

    elif mode == "linear_large":
        log_name = (
            f"{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}"
            + f"_lr{args.lr}_opt{args.opt}_wd{args.wd}_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_ss{args.sample_size}_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
        )
        base_log_name = (
            f"{args.data}_step{args.step}_ts{args.test_size}_w{args.hidden_dim}_bs{args.batch_size}"
            + f"_lr{args.lr}_opt{args.opt}_wd{args.wd}_mom{args.mom}_nl{args.nl}_regNone"
            + f"_e{args.epochs}_seed{args.seed}"
        )

    elif mode == "conv":
        base_log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_regNone_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )

    # ================ transfer learning =================
    elif mode == "transfer":
        base_log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_regNone_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )
    
    elif mode == 'pretrain':
        base_log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_regNone_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )

    elif mode == "finetune":
        reg_name = str(reg_name)
        # add iterative
        if ("spectral" in args.reg or "eig-ub" in args.reg) and args.iterative:
            reg_name = reg_name.replace("spectral", "spectral-iterative")
            reg_name = reg_name.replace("eig-ub", "eig-ub-iterative")

        base_log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}"
            + f"_mom{args.mom}_reg['None']_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_m{args.model}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}"
            + f"_mom{args.mom}_lam{args.lam}_alpha{args.alpha}_beta{args.beta}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )

    # ================ contrastive ====================
    elif mode == "barlow":
        base_log_name = (
            f"{args.data}_barlow_m{args.model}_p{args.projector}_l{args.lambd}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_regNone_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_barlow_m{args.model}_p{args.projector}_l{args.lambd}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )

    elif mode == "simclr":
        base_log_name = (
            f"{args.data}_simclr_m{args.model}_tem{args.temperature}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_regNone_e{args.epochs}_seed{args.seed}"
        )
        log_name = (
            f"{args.data}_simclr_m{args.model}_tem{args.temperature}_bs{args.batch_size}_lr{args.lr}_opt{args.opt}_wd{args.wd}"
            + f"_mom{args.mom}_nl{args.nl}_lam{args.lam}_reg{reg_name}"
            + f"_e{args.epochs}_b{args.burnin}_seed{args.seed}_rf{args.reg_freq}"
            + (f"_ru{args.reg_freq_update}" if args.reg_freq_update is not None else "")
            + (f"_ml{args.max_layer}" if args.max_layer is not None else "")
        )
    else:
        raise NotImplementedError()

    return log_name, base_log_name
