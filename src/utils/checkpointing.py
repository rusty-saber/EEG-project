"""
Checkpoint saving and loading utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: DictConfig,
    path: Union[str, Path],
    scheduler: Optional[Any] = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'val_pearson_mean': 0.75})
        config: Training configuration
        path: Output path for checkpoint
        scheduler: Optional learning rate scheduler
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': OmegaConf.to_container(config, resolve=True),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        model: Optional model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        
    Returns:
        Dictionary containing checkpoint data (epoch, metrics, config)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'config': checkpoint.get('config', {}),
    }


def get_best_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Find the best checkpoint in a directory (looks for 'best.pt')."""
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / 'best.pt'
    
    if best_path.exists():
        return best_path
    
    # Fallback: find checkpoint with highest epoch number
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    if not checkpoints:
        return None
    
    # Sort by epoch number if format is *_epoch{N}.pt
    def get_epoch(p: Path) -> int:
        name = p.stem
        if 'epoch' in name:
            try:
                return int(name.split('epoch')[-1])
            except ValueError:
                return 0
        return 0
    
    return max(checkpoints, key=get_epoch)


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
