#!/usr/bin/env python
"""
MVP (Minimum Viable Pipeline) script.
Quick end-to-end test with 5 subjects and minimal epochs.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Run MVP experiment")
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                       help='Directory with raw data (or to download to)')
    parser.add_argument('--output_dir', type=str, default='./outputs/mvp',
                       help='Output directory')
    parser.add_argument('--num_subjects', type=int, default=5,
                       help='Number of subjects for MVP')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip data download')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='Skip preprocessing')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto-detected if not specified)')
    
    return parser.parse_args()


def run_mvp():
    args = parse_args()
    
    start_time = time.time()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path(args.data_dir)
    processed_dir = output_dir / 'processed'
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("ANTIGRAVITY MVP - Quick End-to-End Test")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Subjects: {args.num_subjects}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Step 1: Download (optional)
    if not args.skip_download:
        print("\n[1/4] Downloading data...")
        from src.data.download import download_physionet
        download_physionet(
            output_dir=str(raw_dir),
            num_subjects=args.num_subjects,
        )
    else:
        print("\n[1/4] Skipping download")
    
    # Step 2: Preprocess
    if not args.skip_preprocess:
        print("\n[2/4] Preprocessing data...")
        from src.data.preprocessing import preprocess_subject, save_preprocessed
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        input_channels = ['Fz', 'Cz', 'Pz', 'Oz']
        target_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        
        preprocess_config = {
            'target_fs': 200,
            'segment_length_sec': 10.0,
            'segment_overlap_sec': 5.0,
            'filter_low': 0.5,
            'filter_high': 45.0,
            'artifact_threshold': 100.0,
        }
        
        # Find and process EDF files
        
        for i in range(1, args.num_subjects + 1):
            subject_id = f"S{i:03d}"
            # Look for EDF files
            edf_files = list(raw_dir.glob(f"**/*{subject_id}*/*.edf")) + \
                       list(raw_dir.glob(f"**/{subject_id}*R*.edf"))
            
            if not edf_files:
                print(f"  No files found for {subject_id}, creating synthetic data...")
                # Create synthetic data for testing
                import numpy as np
                np.random.seed(i)
                n_segments = 10
                n_samples = 2000  # 10s @ 200Hz
                
                data = {
                    'input_eeg': np.random.randn(n_segments, len(input_channels), n_samples).astype(np.float32),
                    'target_eeg': np.random.randn(n_segments, len(target_channels), n_samples).astype(np.float32),
                    'valid_mask': np.ones(n_segments, dtype=bool),
                }
                save_preprocessed(data, str(processed_dir / f"{subject_id}.npz"))
            else:
                try:
                    from tqdm import tqdm
                    all_input = []
                    all_target = []
                    all_valid = []
                    
                    for edf_path in edf_files[:2]:  # Limit for speed
                        result = preprocess_subject(
                            str(edf_path),
                            input_channels,
                            target_channels,
                            preprocess_config
                        )
                        all_input.append(result['input_eeg'])
                        all_target.append(result['target_eeg'])
                        all_valid.append(result['valid_mask'])
                    
                    import numpy as np
                    data = {
                        'input_eeg': np.concatenate(all_input, axis=0),
                        'target_eeg': np.concatenate(all_target, axis=0),
                        'valid_mask': np.concatenate(all_valid, axis=0),
                    }
                    save_preprocessed(data, str(processed_dir / f"{subject_id}.npz"))
                    print(f"  {subject_id}: {data['valid_mask'].sum()} valid segments")
                except Exception as e:
                    print(f"  {subject_id}: Error - {e}, using synthetic data")
                    import numpy as np
                    np.random.seed(i)
                    data = {
                        'input_eeg': np.random.randn(10, 4, 2000).astype(np.float32),
                        'target_eeg': np.random.randn(10, 8, 2000).astype(np.float32),
                        'valid_mask': np.ones(10, dtype=bool),
                    }
                    save_preprocessed(data, str(processed_dir / f"{subject_id}.npz"))
    else:
        print("\n[2/4] Skipping preprocessing")
    
    # Step 3: Train
    print("\n[3/4] Training model...")
    
    from src.utils.config import load_config
    from src.data.dataset import EEGChannelExpansionDataset
    from src.models.full_model import ChannelExpansionModel
    from src.training.losses import CompositeLoss
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Simple training without full trainer
    input_channels = ['Fz', 'Cz', 'Pz', 'Oz']
    target_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
    
    # Split subjects
    all_subjects = [f"S{i:03d}" for i in range(1, args.num_subjects + 1)]
    train_subjects = all_subjects[:-1]
    val_subjects = all_subjects[-1:]
    
    # Use pre-existing processed data directory if available
    data_path = Path('./data/processed')
    if not data_path.exists():
        data_path = processed_dir
    
    # Create datasets - use_all_segments=True to avoid artifact mask issues
    train_dataset = EEGChannelExpansionDataset(
        data_dir=data_path,
        subject_ids=train_subjects,
        input_channels=input_channels,
        target_channels=target_channels,
        use_all_segments=True,  # Use all segments for MVP
    )
    
    val_dataset = EEGChannelExpansionDataset(
        data_dir=data_path,
        subject_ids=val_subjects,
        input_channels=input_channels,
        target_channels=target_channels,
        use_all_segments=True,  # Use all segments for MVP
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create model
    model = ChannelExpansionModel(
        freeze_encoder=True,
        dropout=0.1,
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = CompositeLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Training
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_eeg = batch['input_eeg'].to(device)
            target_eeg = batch['target_eeg'].to(device)
            input_pos = batch['input_positions'].to(device)
            target_pos = batch['target_positions'].to(device)
            
            optimizer.zero_grad()
            output = model(input_eeg, input_pos, target_pos)
            loss_dict = criterion(output['output'], target_eeg)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_eeg = batch['input_eeg'].to(device)
                target_eeg = batch['target_eeg'].to(device)
                input_pos = batch['input_positions'].to(device)
                target_pos = batch['target_positions'].to(device)
                
                output = model(input_eeg, input_pos, target_pos)
                loss_dict = criterion(output['output'], target_eeg)
                val_loss += loss_dict['loss'].item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"  Epoch {epoch+1}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}")
    
    # Save model
    checkpoint_path = output_dir / 'mvp_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, checkpoint_path)
    
    # Step 4: Evaluate and visualize
    print("\n[4/4] Generating plots...")
    
    from src.evaluation.metrics import compute_all_metrics, compute_per_channel_metrics
    from src.evaluation.visualizations import (
        plot_reconstruction, plot_loss_curves, plot_channel_metrics
    )
    
    # Collect predictions
    all_preds = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_eeg = batch['input_eeg'].to(device)
            target_eeg = batch['target_eeg'].to(device)
            input_pos = batch['input_positions'].to(device)
            target_pos = batch['target_positions'].to(device)
            
            output = model(input_eeg, input_pos, target_pos)
            all_preds.append(output['output'].cpu())
            all_targets.append(target_eeg.cpu())
    
    pred = torch.cat(all_preds, dim=0)
    target = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_all_metrics(pred, target, target_channels)
    per_channel = compute_per_channel_metrics(pred, target, target_channels)
    
    # Generate plots
    plot_reconstruction(
        pred[0], target[0], target_channels,
        save_path=str(output_dir / 'reconstruction.png'),
        title="MVP Reconstruction Sample"
    )
    
    plot_loss_curves(
        train_losses, val_losses,
        save_path=str(output_dir / 'loss_curves.png'),
        title="MVP Training Curves"
    )
    
    plot_channel_metrics(
        per_channel, target_channels,
        save_path=str(output_dir / 'metrics.png'),
        title="MVP Per-Channel Metrics"
    )
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("MVP COMPLETE")
    print("=" * 60)
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"\nFinal Results:")
    print(f"  Mean Pearson: {metrics['val_pearson_mean']:.4f}")
    print(f"  Mean SNR: {metrics['val_snr_mean']:.2f} dB")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - reconstruction.png")
    print(f"  - loss_curves.png")
    print(f"  - metrics.png")
    print(f"  - mvp_model.pt")


if __name__ == '__main__':
    run_mvp()
