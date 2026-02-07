"""
Data augmentation transforms for EEG.
"""

from typing import Tuple

import torch
import numpy as np


def add_gaussian_noise(
    eeg: torch.Tensor,
    noise_std: float = 0.1
) -> torch.Tensor:
    """
    Add Gaussian noise to EEG signal.
    
    Args:
        eeg: EEG tensor of shape (..., channels, samples)
        noise_std: Standard deviation of noise
        
    Returns:
        Noisy EEG with same shape
    """
    noise = torch.randn_like(eeg) * noise_std
    return eeg + noise


def time_shift(
    eeg: torch.Tensor,
    max_shift: int = 50
) -> torch.Tensor:
    """
    Apply random circular time shift.
    
    Args:
        eeg: EEG tensor of shape (..., channels, samples)
        max_shift: Maximum shift in samples
        
    Returns:
        Shifted EEG
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(eeg, shifts=shift, dims=-1)


def amplitude_scale(
    eeg: torch.Tensor,
    min_scale: float = 0.8,
    max_scale: float = 1.2
) -> torch.Tensor:
    """
    Apply random amplitude scaling.
    
    Args:
        eeg: EEG tensor of shape (..., channels, samples)
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        
    Returns:
        Scaled EEG
    """
    scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    return eeg * scale


def channel_dropout(
    eeg: torch.Tensor,
    dropout_prob: float = 0.1
) -> torch.Tensor:
    """
    Randomly zero out entire channels.
    
    Args:
        eeg: EEG tensor of shape (channels, samples) or (batch, channels, samples)
        dropout_prob: Probability of dropping each channel
        
    Returns:
        EEG with some channels zeroed
    """
    if eeg.dim() == 2:
        n_channels = eeg.shape[0]
        mask = torch.rand(n_channels) > dropout_prob
        mask = mask.float().unsqueeze(-1)
    else:
        batch_size, n_channels, _ = eeg.shape
        mask = torch.rand(batch_size, n_channels) > dropout_prob
        mask = mask.float().unsqueeze(-1)
    
    return eeg * mask.to(eeg.device)


class EEGAugmentation:
    """
    Compose multiple EEG augmentations.
    """
    
    def __init__(
        self,
        noise_std: float = 0.05,
        time_shift_max: int = 20,
        amplitude_scale_range: Tuple[float, float] = (0.9, 1.1),
        position_jitter_std: float = 0.01,
        apply_prob: float = 0.5
    ):
        """
        Args:
            noise_std: Std for Gaussian noise
            time_shift_max: Max time shift in samples
            amplitude_scale_range: (min, max) for amplitude scaling
            position_jitter_std: Std for electrode position jitter
            apply_prob: Probability of applying each augmentation
        """
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.amplitude_scale_range = amplitude_scale_range
        self.position_jitter_std = position_jitter_std
        self.apply_prob = apply_prob
    
    def __call__(
        self,
        input_eeg: torch.Tensor,
        target_eeg: torch.Tensor,
        input_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to input/target pair.
        
        Important: We apply the SAME augmentation to both input and target
        to maintain their relationship.
        """
        # Gaussian noise (independent for input and target)
        if torch.rand(1).item() < self.apply_prob:
            input_eeg = add_gaussian_noise(input_eeg, self.noise_std)
            target_eeg = add_gaussian_noise(target_eeg, self.noise_std)
        
        # Time shift (same shift for both)
        if torch.rand(1).item() < self.apply_prob:
            shift = torch.randint(
                -self.time_shift_max, self.time_shift_max + 1, (1,)
            ).item()
            input_eeg = torch.roll(input_eeg, shifts=shift, dims=-1)
            target_eeg = torch.roll(target_eeg, shifts=shift, dims=-1)
        
        # Amplitude scale (same scale for both)
        if torch.rand(1).item() < self.apply_prob:
            scale = torch.empty(1).uniform_(
                self.amplitude_scale_range[0],
                self.amplitude_scale_range[1]
            ).item()
            input_eeg = input_eeg * scale
            target_eeg = target_eeg * scale
        
        # Position jitter
        if self.position_jitter_std > 0 and torch.rand(1).item() < self.apply_prob:
            from ..utils.positions import add_position_jitter
            input_positions = add_position_jitter(
                input_positions, self.position_jitter_std
            )
            target_positions = add_position_jitter(
                target_positions, self.position_jitter_std
            )
        
        return input_eeg, target_eeg, input_positions, target_positions
