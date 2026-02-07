#!/usr/bin/env python
"""
Evaluation script for trained models.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.checkpointing import load_checkpoint
from src.data.dataset import create_dataloaders, get_subject_ids
from src.models.full_model import create_model
from src.evaluation.metrics import compute_all_metrics, compute_per_channel_metrics
from src.evaluation.visualizations import (
    plot_reconstruction,
    plot_channel_metrics,
    plot_spectra_comparison,
    create_metrics_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data/processed',
                       help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as checkpoint)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                       help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_plots', type=int, default=3,
                       help='Number of sample reconstruction plots to generate')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_subjects = []
    all_segments = []
    
    from tqdm import tqdm
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_eeg = batch['input_eeg'].to(device)
        target_eeg = batch['target_eeg'].to(device)
        input_positions = batch['input_positions'].to(device)
        target_positions = batch['target_positions'].to(device)
        
        output = model(input_eeg, input_positions, target_positions)
        pred = output['output']
        
        all_preds.append(pred.cpu())
        all_targets.append(target_eeg.cpu())
        all_subjects.extend(batch['subject_id'])
        all_segments.extend(batch['segment_idx'].tolist())
    
    return {
        'pred': torch.cat(all_preds, dim=0),
        'target': torch.cat(all_targets, dim=0),
        'subject_ids': all_subjects,
        'segment_ids': all_segments,
    }


def main():
    args = parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent / 'eval'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Load config from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = checkpoint['config']
    
    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    # Get evaluation subjects
    dataset_name = config.dataset.name
    subjects = get_subject_ids(args.split, dataset_name)
    print(f"Evaluating on {len(subjects)} {args.split} subjects")
    
    # Create dataloader
    loaders = create_dataloaders(
        data_dir=args.data_dir,
        train_subjects=[],  # Not needed for eval
        val_subjects=subjects if args.split == 'val' else [],
        test_subjects=subjects if args.split == 'test' else None,
        input_channels=config.channels.input,
        target_channels=config.channels.target,
        batch_size=args.batch_size,
        num_workers=4,
        augment_train=False,
    )
    
    dataloader = loaders[args.split]
    
    # Run evaluation
    results = evaluate(model, dataloader, args.device)
    
    pred = results['pred']
    target = results['target']
    channel_names = config.channels.target
    
    print(f"\nEvaluated {pred.shape[0]} samples")
    
    # Compute metrics
    metrics = compute_all_metrics(pred, target, channel_names)
    per_channel = compute_per_channel_metrics(pred, target, channel_names)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Pearson Correlation: {metrics['val_pearson_mean']:.4f} ± {metrics['val_pearson_std']:.4f}")
    print(f"Mean SNR: {metrics['val_snr_mean']:.2f} ± {metrics['val_snr_std']:.2f} dB")
    print(f"Mean Spectral Similarity: {metrics['val_spectral_mean']:.4f}")
    print(f"Topographic Correlation: {metrics['val_topo_corr']:.4f}")
    
    print("\nPer-channel Pearson:")
    for i, ch in enumerate(channel_names):
        print(f"  {ch}: {per_channel['pearson_per_channel'][i]:.4f}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Create markdown table
    create_metrics_table(
        metrics, channel_names,
        save_path=str(output_dir / 'results.md')
    )
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Reconstruction plots
    for i in range(min(args.num_plots, len(pred))):
        plot_reconstruction(
            pred[i], target[i], channel_names,
            save_path=str(output_dir / f'reconstruction_{i+1}.png'),
            title=f"Sample {i+1} Reconstruction"
        )
    
    # 2. Channel metrics
    plot_channel_metrics(
        per_channel, channel_names,
        save_path=str(output_dir / 'channel_metrics.png'),
        title="Per-Channel Performance"
    )
    
    # 3. Spectra comparison
    for i, ch in enumerate(channel_names[:2]):  # First 2 channels
        plot_spectra_comparison(
            pred[0], target[0],
            channel_idx=i, channel_name=ch,
            save_path=str(output_dir / f'spectra_{ch}.png')
        )
    
    print(f"\nAll results saved to: {output_dir}")
    
    # Return summary for scripting
    return {
        'pearson': metrics['val_pearson_mean'],
        'snr': metrics['val_snr_mean'],
        'spectral': metrics['val_spectral_mean'],
    }


if __name__ == '__main__':
    main()
