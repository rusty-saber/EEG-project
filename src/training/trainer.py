"""
Training loop implementation.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import CompositeLoss, create_loss
from .schedulers import create_scheduler
from ..utils.checkpointing import save_checkpoint, EarlyStopping
from ..utils.logging_utils import Logger
from ..evaluation.metrics import compute_all_metrics


class Trainer:
    """
    Training loop for channel expansion model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        device: str = 'cuda',
    ):
        """
        Initialize trainer.
        
        Args:
            model: Channel expansion model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            logger: Optional logger for metrics
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        # Training settings
        self.epochs = config.training.epochs
        self.batch_size = config.training.batch_size
        self.gradient_accumulation = config.training.get('gradient_accumulation', 1)
        self.mixed_precision = config.training.get('mixed_precision', True)
        self.gradient_clip = config.training.get('gradient_clip_max_norm', None)
        
        # Loss function
        self.criterion = create_loss(config)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer, config, steps_per_epoch=len(train_loader)
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Early stopping
        es_config = config.early_stopping
        self.early_stopping = EarlyStopping(
            patience=es_config.patience,
            min_delta=es_config.get('min_delta', 0.001),
            mode=es_config.mode,
        ) if es_config.get('enabled', True) else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.best_epoch = 0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optimizer
        
        # Check for differential learning rates
        if 'param_groups' in opt_config:
            param_groups = self.model.get_parameter_groups(
                encoder_lr=opt_config.param_groups[0]['lr'],
                expansion_lr=opt_config.param_groups[1]['lr'],
                decoder_lr=opt_config.param_groups[2]['lr'],
            )
        else:
            param_groups = self.model.parameters()
        
        if opt_config.name == 'AdamW':
            return torch.optim.AdamW(
                param_groups,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=tuple(opt_config.get('betas', [0.9, 0.999])),
            )
        elif opt_config.name == 'Adam':
            return torch.optim.Adam(
                param_groups,
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 0),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_time_loss = 0.0
        total_spectral_loss = 0.0
        total_correlation_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_eeg = batch['input_eeg'].to(self.device)
            target_eeg = batch['target_eeg'].to(self.device)
            input_positions = batch['input_positions'].to(self.device)
            target_positions = batch['target_positions'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.mixed_precision):
                output = self.model(
                    input_eeg, input_positions, target_positions
                )
                pred = output['output']
                
                loss_dict = self.criterion(pred, target_eeg, return_components=True)
                loss = loss_dict['loss'] / self.gradient_accumulation
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.scaler is not None:
                    if self.gradient_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track losses
            total_loss += loss.item() * self.gradient_accumulation
            total_time_loss += loss_dict['time_loss'].item()
            total_spectral_loss += loss_dict['spectral_loss'].item()
            total_correlation_loss += loss_dict['correlation_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.scheduler.get_last_lr()[0],
            })
            
            # Log to logger
            if self.logger and self.global_step % self.config.logging.log_every_n_steps == 0:
                self.logger.log_scalars({
                    'train/loss': loss.item() * self.gradient_accumulation,
                    'train/time_loss': loss_dict['time_loss'].item(),
                    'train/spectral_loss': loss_dict['spectral_loss'].item(),
                    'train/correlation_loss': loss_dict['correlation_loss'].item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                }, step=self.global_step)
        
        return {
            'train_loss': total_loss / num_batches,
            'train_time_loss': total_time_loss / num_batches,
            'train_spectral_loss': total_spectral_loss / num_batches,
            'train_correlation_loss': total_correlation_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            input_eeg = batch['input_eeg'].to(self.device)
            target_eeg = batch['target_eeg'].to(self.device)
            input_positions = batch['input_positions'].to(self.device)
            target_positions = batch['target_positions'].to(self.device)
            
            with autocast(enabled=self.mixed_precision):
                output = self.model(
                    input_eeg, input_positions, target_positions
                )
                pred = output['output']
                
                loss_dict = self.criterion(pred, target_eeg)
                total_loss += loss_dict['loss'].item()
            
            all_preds.append(pred.cpu())
            all_targets.append(target_eeg.cpu())
            num_batches += 1
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = compute_all_metrics(all_preds, all_targets)
        metrics['val_loss'] = total_loss / num_batches
        
        return metrics
    
    def train(self) -> Dict[str, float]:
        """
        Full training loop.
        
        Returns:
            Best metrics achieved during training
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters()}")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Merge metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Print metrics
            print(f"\nEpoch {self.current_epoch}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Pearson: {val_metrics['val_pearson_mean']:.4f}")
            print(f"  Val SNR: {val_metrics['val_snr_mean']:.2f} dB")
            
            # Log to logger
            if self.logger:
                self.logger.log_scalars(all_metrics, step=self.current_epoch)
            
            # Check for best model
            monitor_metric = val_metrics[self.config.early_stopping.metric]
            
            if self.best_metric is None:
                is_best = True
            elif self.config.early_stopping.mode == 'max':
                is_best = monitor_metric > self.best_metric
            else:
                is_best = monitor_metric < self.best_metric
            
            if is_best:
                self.best_metric = monitor_metric
                self.best_epoch = self.current_epoch
                
                # Save best checkpoint
                save_checkpoint(
                    self.model, self.optimizer, self.current_epoch,
                    all_metrics, self.config,
                    self.checkpoint_dir / 'best.pt',
                    self.scheduler
                )
                print(f"  New best! {self.config.early_stopping.metric}: {monitor_metric:.4f}")
            
            # Save periodic checkpoint
            if self.current_epoch % self.config.logging.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.current_epoch,
                    all_metrics, self.config,
                    self.checkpoint_dir / f'epoch_{self.current_epoch}.pt',
                    self.scheduler
                )
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(monitor_metric):
                    print(f"\nEarly stopping at epoch {self.current_epoch}")
                    print(f"Best {self.config.early_stopping.metric}: {self.best_metric:.4f} at epoch {self.best_epoch}")
                    break
        
        # Save final checkpoint
        save_checkpoint(
            self.model, self.optimizer, self.current_epoch,
            all_metrics, self.config,
            self.checkpoint_dir / 'last.pt',
            self.scheduler
        )
        
        if self.logger:
            self.logger.close()
        
        return {'best_metric': self.best_metric, 'best_epoch': self.best_epoch}
