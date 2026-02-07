#!/usr/bin/env python
"""
Main training script for channel expansion.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_experiment_name, print_config
from src.utils.logging_utils import setup_logger
from src.data.dataset import create_dataloaders, get_subject_ids
from src.models.full_model import create_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train EEG Channel Expansion Model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (or comma-separated list)')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='Training stage (1=frozen encoder, 2=fine-tune)')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    # Config overrides
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config_paths = [p.strip() for p in args.config.split(',')]
    overrides = {}
    
    if args.epochs:
        overrides['training.epochs'] = args.epochs
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.lr:
        overrides['optimizer.lr'] = args.lr
    if args.device:
        overrides['device'] = args.device
    
    config = load_config(config_paths, overrides if overrides else None)
    
    # Set device
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experiment name
    experiment_name = args.experiment_name or get_experiment_name(config)
    
    # Output directories
    output_dir = Path(args.output_dir) / experiment_name
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config paths
    config.paths.checkpoint_dir = str(checkpoint_dir)
    config.paths.log_dir = str(log_dir)
    
    print(f"Experiment: {experiment_name}")
    print(f"Output directory: {output_dir}")
    print_config(config)
    
    # Setup logger
    logger = setup_logger(
        experiment_name=experiment_name,
        log_dir=str(log_dir),
        use_wandb=not args.no_wandb,
        config=config,
    )
    
    # Get subject splits
    dataset_name = config.dataset.name
    train_subjects = get_subject_ids('train', dataset_name)
    val_subjects = get_subject_ids('val', dataset_name)
    
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    
    # Create dataloaders
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        input_channels=config.channels.input,
        target_channels=config.channels.target,
        batch_size=config.training.batch_size,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        augment_train=config.augmentation.get('enabled', True),
    )
    
    # Create model
    model = create_model(config)
    
    print(f"\nModel parameters:")
    for name, count in model.count_parameters().items():
        print(f"  {name}: {count:,}")
    
    # Resume if specified
    if args.resume:
        from src.utils.checkpointing import load_checkpoint
        checkpoint = load_checkpoint(args.resume, model, map_location=device)
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=config,
        logger=logger,
        device=device,
    )
    
    # Train
    print("\nStarting training...")
    result = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Best {config.early_stopping.metric}: {result['best_metric']:.4f}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
