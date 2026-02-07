"""
Visualization functions for EEG reconstruction results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_reconstruction(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: List[str],
    sample_rate: int = 200,
    time_range: Tuple[float, float] = (0, 2),
    save_path: Optional[str] = None,
    title: str = "EEG Reconstruction",
) -> plt.Figure:
    """
    Plot predicted vs target EEG for selected channels.
    
    Args:
        pred: (channels, samples) or (batch, channels, samples)
        target: (channels, samples) or (batch, channels, samples)
        channel_names: List of channel names
        sample_rate: Sampling rate in Hz
        time_range: Time range to plot in seconds
        save_path: Optional path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    # Take first batch if batched
    if pred.dim() == 3:
        pred = pred[0]
        target = target[0]
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    n_channels = min(4, len(channel_names))  # Show at most 4 channels
    
    # Time axis
    start_sample = int(time_range[0] * sample_rate)
    end_sample = int(time_range[1] * sample_rate)
    time = np.arange(start_sample, end_sample) / sample_rate
    
    # Create figure
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 3 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        p = pred[i, start_sample:end_sample]
        t = target[i, start_sample:end_sample]
        
        ax.plot(time, t, 'b-', label='Target', alpha=0.8, linewidth=1)
        ax.plot(time, p, 'r-', label='Predicted', alpha=0.8, linewidth=1)
        
        ax.set_ylabel(channel_names[i])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Compute correlation for this channel
        corr = np.corrcoef(p, t)[0, 1]
        ax.set_title(f"{channel_names[i]} (r = {corr:.3f})")
    
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark minimum validation loss
    min_idx = np.argmin(val_losses)
    ax.axvline(min_idx + 1, color='g', linestyle='--', alpha=0.5, label=f'Best: epoch {min_idx + 1}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_channel_metrics(
    metrics_per_channel: Dict[str, np.ndarray],
    channel_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Per-Channel Metrics",
) -> plt.Figure:
    """
    Bar chart of metrics per channel.
    
    Args:
        metrics_per_channel: Dict with 'pearson_per_channel', 'snr_per_channel'
        channel_names: List of channel names
        save_path: Optional path to save figure
        title: Figure title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(channel_names))
    width = 0.6
    
    # Pearson correlation
    pearson = metrics_per_channel['pearson_per_channel']
    bars1 = axes[0].bar(x, pearson, width, color='steelblue', edgecolor='black')
    axes[0].set_ylabel("Pearson Correlation")
    axes[0].set_title("Per-Channel Correlation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_names, rotation=45)
    axes[0].axhline(0.75, color='r', linestyle='--', label='Target (0.75)')
    axes[0].axhline(pearson.mean(), color='g', linestyle='-', label=f'Mean ({pearson.mean():.3f})')
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, pearson):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # SNR
    snr = metrics_per_channel['snr_per_channel']
    bars2 = axes[1].bar(x, snr, width, color='coral', edgecolor='black')
    axes[1].set_ylabel("SNR (dB)")
    axes[1].set_title("Per-Channel SNR")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_names, rotation=45)
    axes[1].axhline(18, color='r', linestyle='--', label='Target (18 dB)')
    axes[1].axhline(snr.mean(), color='g', linestyle='-', label=f'Mean ({snr.mean():.1f} dB)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, snr):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_spectra_comparison(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_idx: int = 0,
    channel_name: str = "Channel",
    sample_rate: int = 200,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare power spectra of predicted and target signals.
    
    Args:
        pred: (samples,) or (channels, samples) predicted signal
        target: (samples,) or (channels, samples) target signal
        channel_idx: Which channel to plot
        channel_name: Name of channel
        sample_rate: Sampling rate
        save_path: Optional save path
        
    Returns:
        Matplotlib figure
    """
    from scipy import signal as sig
    
    if pred.dim() > 1:
        pred = pred[channel_idx]
        target = target[channel_idx]
    
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Compute PSDs
    freq, psd_pred = sig.welch(pred, fs=sample_rate, nperseg=sample_rate * 2)
    _, psd_target = sig.welch(target, fs=sample_rate, nperseg=sample_rate * 2)
    
    # Filter to relevant frequencies
    mask = (freq >= 0.5) & (freq <= 45)
    freq = freq[mask]
    psd_pred = psd_pred[mask]
    psd_target = psd_target[mask]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.semilogy(freq, psd_target, 'b-', label='Target', linewidth=2, alpha=0.8)
    ax.semilogy(freq, psd_pred, 'r-', label='Predicted', linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(f"Power Spectrum Comparison - {channel_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 45])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_metrics_table(
    metrics: Dict[str, float],
    channel_names: List[str],
    save_path: Optional[str] = None,
) -> str:
    """
    Create a markdown table of metrics.
    
    Args:
        metrics: Dictionary of metrics
        channel_names: List of channel names
        save_path: Optional path to save markdown file
        
    Returns:
        Markdown table string
    """
    # Header
    header = "| Metric |" + " | ".join(channel_names) + " | Mean |"
    separator = "|" + "|".join(["---"] * (len(channel_names) + 2)) + "|"
    
    # Pearson row
    pearson_vals = [metrics.get(f'val_pearson_{ch}', 0.0) for ch in channel_names]
    pearson_mean = metrics.get('val_pearson_mean', np.mean(pearson_vals))
    pearson_row = "| Pearson |" + " | ".join([f"{v:.3f}" for v in pearson_vals]) + f" | **{pearson_mean:.3f}** |"
    
    # SNR row
    snr_vals = [metrics.get(f'val_snr_{ch}', 0.0) for ch in channel_names]
    snr_mean = metrics.get('val_snr_mean', np.mean(snr_vals))
    snr_row = "| SNR (dB) |" + " | ".join([f"{v:.1f}" for v in snr_vals]) + f" | **{snr_mean:.1f}** |"
    
    table = "\n".join([header, separator, pearson_row, snr_row])
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write("# Evaluation Results\n\n")
            f.write(table)
            f.write("\n\n## Summary\n")
            f.write(f"- Mean Pearson: {pearson_mean:.4f}\n")
            f.write(f"- Mean SNR: {snr_mean:.2f} dB\n")
            f.write(f"- Spectral Similarity: {metrics.get('val_spectral_mean', 0):.4f}\n")
        print(f"Saved metrics table to {save_path}")
    
    return table


def plot_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: List[str],
    output_dir: str,
    sample_idx: int = 0,
) -> None:
    """
    Generate all standard plots and save to output directory.
    
    Args:
        pred: (batch, channels, samples) predictions
        target: (batch, channels, samples) targets
        channel_names: Channel names
        output_dir: Directory to save plots
        sample_idx: Which sample to use for waveform plots
    """
    from .metrics import compute_per_channel_metrics
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    per_channel = compute_per_channel_metrics(pred, target, channel_names)
    
    # 1. Reconstruction overlay
    plot_reconstruction(
        pred[sample_idx], target[sample_idx], channel_names,
        save_path=str(output_dir / "reconstruction.png"),
        title="Sample Reconstruction"
    )
    
    # 2. Channel metrics bar chart
    plot_channel_metrics(
        per_channel, channel_names,
        save_path=str(output_dir / "channel_metrics.png"),
        title="Per-Channel Metrics"
    )
    
    # 3. Spectra comparison for first channel
    plot_spectra_comparison(
        pred[sample_idx], target[sample_idx],
        channel_idx=0, channel_name=channel_names[0],
        save_path=str(output_dir / "spectra_comparison.png")
    )
    
    print(f"All plots saved to {output_dir}")
