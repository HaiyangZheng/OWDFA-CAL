import os
import torch
import inspect

from datetime import datetime
from loguru import logger

def init_experiment(args):

    # The experiment root directory is exp_root / runner_name
    if args.runner_name is None:
        raise ValueError("Need to specify the runner name")
    else:
        root_dir = os.path.join(args.exp_root, args.runner_name)

    # Create the root directory if it does not exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Raise an error if exp_name is not provided
    if args.exp_name is None:
        raise ValueError("Need to specify the experiment name")
    else:
        # Construct a unique experiment ID as the log directory name, roughly in the format: exp_name_(DD.MM.YYYY_|_SS.mmm)
        now = '{}_({:02d}.{:02d}.{}_|_'.format(
            args.exp_name,
            datetime.now().day,
            datetime.now().month,
            datetime.now().year
        ) + datetime.now().strftime("%S.%f")[:-3] + ')'
        log_dir = os.path.join(root_dir, now)

        # Ensure uniqueness of the log directory
        while os.path.exists(log_dir):
            now = '{}_({:02d}.{:02d}.{}_|_'.format(
                args.exp_name,
                datetime.now().day,
                datetime.now().month,
                datetime.now().year
            ) + datetime.now().strftime("%S.%f")[:-3] + ')'
            log_dir = os.path.join(root_dir, now)

    # Create the experiment log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Write logs to log.txt under the experiment directory
    logger.add(os.path.join(log_dir, 'log.txt'))

    args.log_dir = log_dir

    # Create a checkpoints subfolder under the log directory
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    # Record the model directory and main model file path (model.pt) in args
    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    logger.info(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    logger.info(args.runner_name)
    logger.info(args)

    return args
