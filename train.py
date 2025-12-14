import argparse
import os
import random

import numpy as np
import wandb
from loguru import logger

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

import CAL.algorithm as algorithm
from dataset.owdfa_protocols import get_classes_from_protocol
from dataset.get_dataset import get_dataset
from dataset.transforms import WS_ViewGenerator, create_data_transforms_ws
from utils.general_utils import init_experiment

def init_seed_torch(seed=42):
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch."""
    
    # Python & OS
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # NumPy
    np.random.seed(seed)

    # PyTorch General
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch Deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main():

    parser = argparse.ArgumentParser(description="Training configuration for CAL-OWDFA")

    # Data & Input settings
    parser.add_argument("--dataset_root", type=str, default="", help="Root directory of the dataset")
    parser.add_argument("--predictor_path", type=str, default="", help="Path to auxiliary predictor")
    parser.add_argument("--image_size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406], help="Normalization mean for RGB channels")
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225], help="Normalization std for RGB channels")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size per iteration")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--print_interval", type=int, default=10, help="Interval (in steps) for printing training logs")
    parser.add_argument("--val_interval", type=int, default=1, help="Interval (in epochs) for running validation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loading")
    parser.add_argument("--seed", type=int, default=0,help="Random seed for reproducibility")
    parser.add_argument("--prop_train_labels", type=float, default=0.75, help="Proportion of labeled samples in the training set")

    # Method settings
    parser.add_argument("--method_name", type=str, default="CAL_OWDFA", help="Name of the training method to use")
    parser.add_argument("--ccr_weight", type=float, default=0.2, help="Weight of CCR (confidence-aware consistency regularization) loss")
    parser.add_argument("--student_temp", type=float, default=1.0, help="Temperature for student predictions")
    parser.add_argument("--teacher_temp", type=float, default=0.1, help="Final temperature for teacher predictions")

    # Pseudo-labeling / ACR settings
    parser.add_argument("--pseudo_epohcs", type=int, default=5, help="Epoch to start pseudo-label supervision")
    parser.add_argument("--gamma", type=float, default=0.9, help="Confidence threshold for known-class pseudo labels")

    # Prototype pruning / unknown-K estimation
    parser.add_argument("--unlabeled_usage_coverage", type=float, default=0.9544, help="Target cumulative usage coverage for unlabeled prototypes")
    parser.add_argument("--enable_proto_pruning", action="store_true", help="Enable dynamic unlabeled prototype pruning (unknown-K setting)")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="CAL_Classifier", help="Model architecture name")
    parser.add_argument("--model_encoder", type=str, default="resnet50", help="Backbone encoder architecture")
    parser.add_argument("--model_num_classes", type=int, default=41, help="Total number of classifier prototypes/classes")
    parser.add_argument("--model_drop_rate", type=float, default=0.2, help="Dropout rate used in the model")
    parser.add_argument("--model_pretrained", type=bool, default=True, help="Whether to use ImageNet-pretrained backbone weights")
    parser.add_argument("--model_is_feat", type=bool, default=False, help="Whether the model outputs features instead of logits")
    parser.add_argument("--model_neck", type=str, default="bnneck", help="Type of neck layer used after the encoder")
    parser.add_argument("--model_resume", type=str, default=None, help="Path to resume model checkpoint (if any)")

    # Optimizer configuration
    parser.add_argument("--optimizer_name", type=str, default="Adam", help="Optimizer type")
    parser.add_argument("--optimizer_lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--optimizer_weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")

    # Scheduler configuration
    parser.add_argument("--scheduler_name", type=str, default="StepLR", help="Learning rate scheduler type")
    parser.add_argument("--scheduler_step_size", type=int, default=10, help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler_gamma", type=float, default=0.05, help="Decay factor for learning rate scheduler")

    # Experiment & logging
    parser.add_argument("--exp_root", type=str, default="exp", help="Root directory for all experiments")
    parser.add_argument("--runner_name", type=str, default="Test", help="Experiment group / runner name")
    parser.add_argument("--exp_name", type=str, default="CAL_protocol1", help="Experiment name (used for logging directory)")
    parser.add_argument("--protocol", type=int, default=1, help="Dataset protocol or split setting")

    # Weights & Biases (optional)
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API key")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity / user name")

    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on the test set")
    parser.add_argument("--eval_ckpt", type=str, default=None, help="Path to checkpoint for evaluation")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Training Protocol Configuration
    #   - Select known / train classes according to protocol
    #   - Optionally expand classifier capacity for unknown-K setting
    # ---------------------------------------------------------
    known_classes, train_classes = get_classes_from_protocol(args.protocol)

    args.known_classes = known_classes
    args.train_classes = train_classes

    logger.info(f"Protocol {args.protocol}, with known classes {known_classes}, all classes {train_classes}")

    if args.enable_proto_pruning:
        args.model_num_classes = 10 * len(known_classes)

    # ---------------------------------------------------------
    # Training Configurations
    # ---------------------------------------------------------

    # Initialize random seeds
    init_seed_torch(args.seed)
    pl.seed_everything(args.seed, workers=True)

    # Build nested model configuration (args.model)
    args.model = argparse.Namespace()
    args.model.name = args.model_name
    args.model.resume = args.model_resume
    args.model.params = argparse.Namespace(
        encoder=args.model_encoder,
        num_classes=args.model_num_classes,
        drop_rate=args.model_drop_rate,
        pretrained=args.model_pretrained,
        is_feat=args.model_is_feat,
        neck=args.model_neck
    )

    # Build nested optimizer configuration (args.optimizer)
    args.optimizer = argparse.Namespace()
    args.optimizer.name = args.optimizer_name
    args.optimizer.params = argparse.Namespace(
        lr=args.optimizer_lr,
        weight_decay=args.optimizer_weight_decay
    )

    # Build nested scheduler configuration (args.scheduler)
    args.scheduler = argparse.Namespace()
    args.scheduler.name = args.scheduler_name

    if args.scheduler_name == "CosineAnnealingLR":
        args.scheduler.params = argparse.Namespace(
            T_max=args.scheduler_T_max,
            eta_min=args.scheduler_eta_min
        )
    elif args.scheduler_name == "StepLR":
        args.scheduler.params = argparse.Namespace(
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
    elif args.scheduler_name == "None":
        args.scheduler.params = None  # 不需要参数
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler_name}")

    # Build nested training configuration (args.train)
    args.train = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        print_interval=args.print_interval,
        val_interval=args.val_interval,
        num_workers=args.num_workers
    )

    # Initialize experiment directories, logging, and checkpoints
    init_experiment(args)


    # ---------------------------------------------------------
    # WANDB SETUP
    # ---------------------------------------------------------
    use_wandb = False # By default, WandB is disabled
    
    # Enable WandB only when required identifiers are explicitly specified
    if args.wandb_project is not None and args.wandb_entity is not None:
        logger.info(f"WandB Enabled. Project: {args.wandb_project}, Entity: {args.wandb_entity}")
        use_wandb = True
        
        if args.wandb_api_key is not None:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity, 
            name=args.exp_name
        )
        
        pl_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name
        )
    else:
        # Fallback: disable WandB and use TensorBoardLogger instead
        logger.info("WandB Disabled. Using TensorBoardLogger.")

        os.environ["WANDB_MODE"] = "disabled"
        pl_logger = TensorBoardLogger(save_dir=args.log_dir, name="lightning_logs")
    
    args.use_wandb=use_wandb



    # ---------------------------------------------------------
    # DATASETS & DATALOADERS
    # ---------------------------------------------------------
    # Transform
    weak_transform, strong_transform = create_data_transforms_ws(args, 'train')
    train_transform = WS_ViewGenerator(weak_transform=weak_transform, strong_transform=strong_transform)
    test_transform = create_data_transforms_ws(args, 'test')

    train_dataset, test_dataset = get_dataset(args, train_transform=train_transform, test_transform=test_transform)

    # Sampler
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # Dataloader
    train_dataloader = DataLoader(train_dataset, num_workers=args.train.num_workers, batch_size=args.train.batch_size, shuffle=False, sampler=sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=128, shuffle=False, pin_memory=False)



    model = algorithm.__dict__[args.method_name](args)

    # Checkpointing: save the best model according to validation accuracy

    # Use the checkpoint directory created by `init_experiment`
    checkpoint_dir = args.model_dir

    best_ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"best_{args.protocol}_{args.seed}",
        monitor="all_acc",
        mode="max",
        save_top_k=1,
        save_last=False,              
        save_on_train_epoch_end=False
    )

    trainer = pl.Trainer(
        accelerator="auto",  # auto-detect GPU/CPU
        devices=1 if torch.cuda.is_available() else None,  # single-device training if CUDA available
        strategy="auto",     # no distributed strategy for single GPU
        min_epochs=1,
        max_epochs=args.train.epochs,
        default_root_dir=args.log_dir,  # root directory for logs/checkpoints (but we set ckpt dir explicitly above)
        callbacks=[best_ckpt_cb],       # register checkpoint callback
        enable_checkpointing=True,      # must be True if you want Lightning to save checkpoints
        num_sanity_val_steps=1,         # run a few validation steps before training to catch errors
        log_every_n_steps=10,           # logging frequency
        logger=pl_logger,
    )

    if args.eval_only:
        assert args.eval_ckpt is not None, "Please specify --eval_ckpt"

        logger.info(f"Evaluating checkpoint: {args.eval_ckpt}")

        trainer.test(
            model=model,
            dataloaders=test_dataloader,
            ckpt_path=args.eval_ckpt
        )
    else:
        trainer.fit(model, train_dataloader, test_dataloader)

    if args.use_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()
