"""Evaluation metrics and visualization utilities."""

from .metrics import (
    compute_all_metrics,
    compute_per_channel_metrics,
    compute_per_subject_metrics,
    pearson_correlation,
    snr_db,
    spectral_similarity,
    topographic_correlation,
)
from .visualizations import (
    plot_reconstruction,
    plot_loss_curves,
    plot_channel_metrics,
    plot_spectra_comparison,
    create_metrics_table,
    plot_metrics,
)

__all__ = [
    'compute_all_metrics',
    'compute_per_channel_metrics',
    'compute_per_subject_metrics',
    'pearson_correlation',
    'snr_db',
    'spectral_similarity',
    'topographic_correlation',
    'plot_reconstruction',
    'plot_loss_curves',
    'plot_channel_metrics',
    'plot_spectra_comparison',
    'create_metrics_table',
    'plot_metrics',
]
