"""
Logging utilities for TensorBoard and WandB.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class Logger:
    """
    Unified logging interface for TensorBoard and WandB.
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        backend: str = 'tensorboard',
        config: Optional[Dict[str, Any]] = None,
        project_name: str = 'antigravity'
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for logs
            experiment_name: Name of the experiment
            backend: 'tensorboard' or 'wandb'
            config: Configuration to log
            project_name: Project name for WandB
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.experiment_name = experiment_name
        self.backend = backend
        self.config = config
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if backend == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        elif backend == 'wandb':
            import wandb
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=str(log_dir)
            )
            self.writer = wandb
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.backend == 'tensorboard':
            self.writer.add_scalar(tag, value, step)
        else:
            self.writer.log({tag: value}, step=step)
    
    def log_scalars(self, scalars: Dict[str, float], step: int) -> None:
        """Log multiple scalar values."""
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """Log an image (expects CHW format)."""
        if self.backend == 'tensorboard':
            self.writer.add_image(tag, image, step)
        else:
            import wandb
            self.writer.log({tag: wandb.Image(image)}, step=step)
    
    def log_figure(self, tag: str, figure, step: int) -> None:
        """Log a matplotlib figure."""
        if self.backend == 'tensorboard':
            self.writer.add_figure(tag, figure, step)
        else:
            import wandb
            self.writer.log({tag: wandb.Image(figure)}, step=step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of values."""
        if self.backend == 'tensorboard':
            self.writer.add_histogram(tag, values, step)
        else:
            import wandb
            self.writer.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def close(self) -> None:
        """Close the logger."""
        if self.backend == 'tensorboard':
            self.writer.close()
        else:
            import wandb
            wandb.finish()


def setup_logger(config) -> Logger:
    """
    Setup logger from configuration.
    
    Args:
        config: OmegaConf configuration with logging settings
        
    Returns:
        Logger instance
    """
    from .config import get_experiment_name
    
    experiment_name = get_experiment_name(config)
    
    return Logger(
        log_dir=config.paths.log_dir,
        experiment_name=experiment_name,
        backend=config.logging.backend,
        config=dict(config),
        project_name=config.logging.project_name
    )
