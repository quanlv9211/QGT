import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import argparse
import random
import logging
import torch.optim as optim
import geoopt


def lr_schedule(epoch):
    # Number of steps (every 50 epochs)
    step = epoch // 50
    step_module = step % 8
    
    if step_module <= 3:
        return 1.0 - step_module * 0.3  # 1.0, 0.7, 0.4, 0.1
    elif step_module <= 7:
        return 0.1 + (step_module - 3) * 0.3  # 0.4, 0.7, 1.0

def lr_schedule_2(epoch):
    # Number of steps (every 50 epochs)
    step = epoch // 50
    step_module = step % 4
    
    if step_module <= 3:
        return 1.0 - step_module * 0.3  # 1.0, 0.7, 0.4, 0.1

def lr_schedule_3(epoch):
    # Number of steps (every 50 epochs)
    step = epoch // 25
    step_module = step % 12
    
    if step_module <= 3:
        return 1.0 - step_module * 0.3  # 1.0, 0.7, 0.4, 0.1
    elif step_module <= 7:
        return 0.1
    elif step_module <= 10:
        return 0.1 + (step_module - 7) * 0.3 # 0.4, 0.7, 1.0
    else:
        return 1.0

def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    print('>> fixed random seed')

def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, float(metric_val)) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

def init_logger(log_dir):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = log_dir + 'result' + '.txt'

    file_handler = logging.FileHandler(log_path,  mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


def find_duplicate_params(param_groups):
    seen = set()
    duplicates = set()

    for i, group in enumerate(param_groups):
        for p in group:
            pid = id(p)
            if pid in seen:
                duplicates.add(pid)
            seen.add(pid)
    
    return duplicates

def print_duplicate_param_names(model, param_groups):
    duplicates = find_duplicate_params(param_groups)
    if not duplicates:
        print("✅ No duplicates found.")
        return
    
    print("❌ Duplicate parameters found:")
    for name, param in model.named_parameters():
        if id(param) in duplicates:
            print(f"  → {name}")

def load_optimizer(args, model):
    if args.using_riemannianAdam:
        optimizer = geoopt.optim.radam.RiemannianAdam(model.parameters(), lr=args.lr,
                                                          weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.model == 'QGT':
        print_duplicate_param_names(model, [model.params1, model.params2, model.params3])
        optimizer = torch.optim.Adam([
            {'params': model.params1, 'weight_decay': args.weight_decay},
            {'params': model.params2, 'weight_decay': args.weight_decay_2},
            {'params': model.params3, 'weight_decay': args.weight_decay_3}
        ],
            lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.gamma)
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_3)
    return optimizer, lr_scheduler
