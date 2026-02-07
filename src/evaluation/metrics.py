"""
Evaluation metrics for EEG reconstruction.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import signal


def pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        pred: Predicted signal
        target: Target signal
        dim: Dimension along which to compute correlation
        
    Returns:
        Correlation coefficient(s)
    """
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)
    
    numerator = (pred_centered * target_centered).sum(dim=dim)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=dim) * (target_centered ** 2).sum(dim=dim)
    )
    
    return numerator / (denominator + 1e-8)


def snr_db(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Compute Signal-to-Noise Ratio in dB.
    
    SNR = 10 * log10(signal_power / noise_power)
    where signal_power = mean(target^2)
          noise_power = mean((pred - target)^2)
    
    Args:
        pred: Predicted signal
        target: Target signal
        dim: Dimension along which to compute SNR
        
    Returns:
        SNR in dB
    """
    signal_power = (target ** 2).mean(dim=dim)
    noise_power = ((pred - target) ** 2).mean(dim=dim)
    
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
    return snr


def spectral_similarity(
    pred: np.ndarray,
    target: np.ndarray,
    fs: int = 200,
    freq_low: float = 0.5,
    freq_high: float = 45.0,
) -> float:
    """
    Compute spectral similarity as correlation of log PSDs.
    
    Args:
        pred: Predicted signal (1D numpy array)
        target: Target signal (1D numpy array)
        fs: Sampling frequency
        freq_low: Lower frequency bound
        freq_high: Upper frequency bound
        
    Returns:
        Spectral similarity (correlation coefficient)
    """
    # Compute PSDs using Welch's method
    nperseg = min(fs * 2, len(pred))
    freq, psd_pred = signal.welch(pred, fs=fs, nperseg=nperseg)
    _, psd_target = signal.welch(target, fs=fs, nperseg=nperseg)
    
    # Select frequency range
    mask = (freq >= freq_low) & (freq <= freq_high)
    
    if mask.sum() < 2:
        return 0.0
    
    log_psd_pred = np.log(psd_pred[mask] + 1e-10)
    log_psd_target = np.log(psd_target[mask] + 1e-10)
    
    # Compute correlation
    corr = np.corrcoef(log_psd_pred, log_psd_target)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def topographic_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Compute topographic correlation (spatial pattern correlation over time).
    
    Args:
        pred: (channels, samples) predicted EEG
        target: (channels, samples) target EEG
        
    Returns:
        Mean topographic correlation
    """
    if pred.dim() == 3:
        # Batch dimension - average over batch
        pred = pred.mean(dim=0)
        target = target.mean(dim=0)
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    correlations = []
    for t in range(pred_np.shape[1]):
        corr = np.corrcoef(pred_np[:, t], target_np[:, t])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    return float(np.mean(correlations)) if correlations else 0.0


def compute_per_channel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for each channel.
    
    Args:
        pred: (batch, channels, samples) or (channels, samples)
        target: (batch, channels, samples) or (channels, samples)
        channel_names: Optional list of channel names
        
    Returns:
        Dictionary with per-channel metrics
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    B, C, T = pred.shape
    
    # Per-channel Pearson correlation (average over batch)
    pearson_per_channel = pearson_correlation(pred, target, dim=-1).mean(dim=0).cpu().numpy()
    
    # Per-channel SNR (average over batch)
    snr_per_channel = snr_db(pred, target, dim=-1).mean(dim=0).cpu().numpy()
    
    # Per-channel spectral similarity
    spectral_per_channel = np.zeros(C)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    for c in range(C):
        similarities = []
        for b in range(B):
            sim = spectral_similarity(pred_np[b, c], target_np[b, c])
            similarities.append(sim)
        spectral_per_channel[c] = np.mean(similarities)
    
    result = {
        'pearson_per_channel': pearson_per_channel,
        'snr_per_channel': snr_per_channel,
        'spectral_per_channel': spectral_per_channel,
    }
    
    # Add named versions if channel names provided
    if channel_names is not None:
        for i, name in enumerate(channel_names):
            result[f'pearson_{name}'] = pearson_per_channel[i]
            result[f'snr_{name}'] = snr_per_channel[i]
            result[f'spectral_{name}'] = spectral_per_channel[i]
    
    return result


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: (batch, channels, samples) predicted EEG
        target: (batch, channels, samples) target EEG
        channel_names: Optional list of channel names
        
    Returns:
        Dictionary with all metrics
    """
    if channel_names is None:
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    
    # Per-channel metrics
    per_channel = compute_per_channel_metrics(pred, target, channel_names)
    
    # Aggregate metrics
    metrics = {
        'val_pearson_mean': float(per_channel['pearson_per_channel'].mean()),
        'val_pearson_std': float(per_channel['pearson_per_channel'].std()),
        'val_snr_mean': float(per_channel['snr_per_channel'].mean()),
        'val_snr_std': float(per_channel['snr_per_channel'].std()),
        'val_spectral_mean': float(per_channel['spectral_per_channel'].mean()),
        'val_spectral_std': float(per_channel['spectral_per_channel'].std()),
    }
    
    # Add per-channel metrics with channel names
    for name in channel_names:
        if f'pearson_{name}' in per_channel:
            metrics[f'val_pearson_{name}'] = float(per_channel[f'pearson_{name}'])
            metrics[f'val_snr_{name}'] = float(per_channel[f'snr_{name}'])
    
    # Topographic correlation
    metrics['val_topo_corr'] = topographic_correlation(pred, target)
    
    return metrics


def compute_per_subject_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    subject_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics grouped by subject.
    
    Args:
        pred: (N, channels, samples) all predictions
        target: (N, channels, samples) all targets
        subject_ids: List of subject IDs for each sample
        
    Returns:
        Dictionary with per-subject metrics
    """
    unique_subjects = list(set(subject_ids))
    per_subject = {}
    
    for subject in unique_subjects:
        # Get indices for this subject
        indices = [i for i, s in enumerate(subject_ids) if s == subject]
        
        # Compute metrics for this subject
        subject_pred = pred[indices]
        subject_target = target[indices]
        
        pearson = pearson_correlation(subject_pred, subject_target, dim=-1).mean()
        snr = snr_db(subject_pred, subject_target, dim=-1).mean()
        
        per_subject[subject] = {
            'pearson': float(pearson),
            'snr': float(snr),
            'n_samples': len(indices),
        }
    
    return per_subject
