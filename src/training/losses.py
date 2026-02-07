"""
Loss functions for EEG channel expansion.
Includes time-domain, spectral, and correlation losses.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDomainLoss(nn.Module):
    """
    Time-domain reconstruction loss.
    Supports MSE, L1, and Smooth L1.
    """
    
    def __init__(self, loss_type: str = 'smooth_l1'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, channels, samples) predicted EEG
            target: (batch, channels, samples) target EEG
            
        Returns:
            Scalar loss value
        """
        return self.loss_fn(pred, target)


class SpectralLoss(nn.Module):
    """
    Spectral domain loss using FFT.
    Computes L1 loss on log power spectra.
    """
    
    def __init__(
        self,
        fs: int = 200,
        freq_low: float = 0.5,
        freq_high: float = 45.0,
        log_scale: bool = True,
    ):
        super().__init__()
        self.fs = fs
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.log_scale = log_scale
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, channels, samples) predicted EEG
            target: (batch, channels, samples) target EEG
            
        Returns:
            Scalar spectral loss
        """
        # Compute FFT
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Power spectra
        pred_power = torch.abs(pred_fft) ** 2
        target_power = torch.abs(target_fft) ** 2
        
        # Frequency bins
        n_samples = pred.shape[-1]
        freqs = torch.fft.rfftfreq(n_samples, 1.0 / self.fs).to(pred.device)
        
        # Select frequency range
        mask = (freqs >= self.freq_low) & (freqs <= self.freq_high)
        pred_power = pred_power[..., mask]
        target_power = target_power[..., mask]
        
        # Log scale
        if self.log_scale:
            pred_power = torch.log(pred_power + 1e-10)
            target_power = torch.log(target_power + 1e-10)
        
        # L1 loss on power spectra
        return F.l1_loss(pred_power, target_power)


class CorrelationLoss(nn.Module):
    """
    Correlation structure preservation loss.
    Encourages predicted channels to have similar inter-channel correlations as target.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (batch, channels, samples) predicted EEG
            target: (batch, channels, samples) target EEG
            
        Returns:
            Scalar correlation loss
        """
        # Compute correlation matrices
        pred_corr = self._batch_corrcoef(pred)
        target_corr = self._batch_corrcoef(target)
        
        # MSE between correlation matrices
        return F.mse_loss(pred_corr, target_corr)
    
    def _batch_corrcoef(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation matrix for each batch element.
        
        Args:
            x: (batch, channels, samples)
            
        Returns:
            corr: (batch, channels, channels)
        """
        # Center
        x = x - x.mean(dim=-1, keepdim=True)
        
        # Compute covariance
        B, C, T = x.shape
        cov = torch.bmm(x, x.transpose(-2, -1)) / (T - 1)
        
        # Normalize to correlation
        std = x.std(dim=-1, keepdim=True)
        std_outer = torch.bmm(std, std.transpose(-2, -1))
        corr = cov / (std_outer + 1e-8)
        
        return corr


class CompositeLoss(nn.Module):
    """
    Composite loss combining time, spectral, and correlation losses.
    """
    
    def __init__(
        self,
        time_domain_weight: float = 1.0,
        spectral_weight: float = 0.5,
        correlation_weight: float = 0.3,
        time_domain_type: str = 'smooth_l1',
        fs: int = 200,
    ):
        super().__init__()
        
        self.time_domain_weight = time_domain_weight
        self.spectral_weight = spectral_weight
        self.correlation_weight = correlation_weight
        
        self.time_loss = TimeDomainLoss(time_domain_type)
        self.spectral_loss = SpectralLoss(fs=fs)
        self.correlation_loss = CorrelationLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            pred: (batch, channels, samples) predicted EEG
            target: (batch, channels, samples) target EEG
            return_components: Whether to return individual loss components
            
        Returns:
            Dict with 'loss' and optionally individual components
        """
        # Compute individual losses
        time_loss = self.time_loss(pred, target)
        spectral_loss = self.spectral_loss(pred, target)
        correlation_loss = self.correlation_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.time_domain_weight * time_loss +
            self.spectral_weight * spectral_loss +
            self.correlation_weight * correlation_loss
        )
        
        result = {'loss': total_loss}
        
        if return_components:
            result.update({
                'time_loss': time_loss,
                'spectral_loss': spectral_loss,
                'correlation_loss': correlation_loss,
            })
        
        return result


def create_loss(config) -> CompositeLoss:
    """Create loss function from config."""
    loss_cfg = config.loss
    return CompositeLoss(
        time_domain_weight=loss_cfg.time_domain_weight,
        spectral_weight=loss_cfg.spectral_weight,
        correlation_weight=loss_cfg.correlation_weight,
        time_domain_type=loss_cfg.get('time_domain_type', 'smooth_l1'),
        fs=config.data.sample_rate,
    )
