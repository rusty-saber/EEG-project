"""
Learning rate schedulers.
"""

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)


class WarmupCosineScheduler:
    """
    Cosine annealing with linear warmup.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        
        # Get base LRs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self) -> None:
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            scale = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * scale
    
    def get_last_lr(self) -> list:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']


def create_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    steps_per_epoch: Optional[int] = None
):
    """
    Create scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        steps_per_epoch: Number of steps per epoch (for step-based scheduling)
    """
    name = config.scheduler.name
    
    if name == 'CosineAnnealingLR':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.get('eta_min', 1e-7),
        )
    
    elif name == 'WarmupCosine':
        warmup_steps = config.scheduler.get('warmup_steps', 500)
        if steps_per_epoch is not None:
            total_steps = config.training.epochs * steps_per_epoch
        else:
            total_steps = config.scheduler.get('total_steps', 10000)
        
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=config.scheduler.get('eta_min', 1e-7),
        )
    
    elif name == 'StepLR':
        return StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.get('gamma', 0.1),
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")
