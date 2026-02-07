#!/usr/bin/env python
"""
Preprocessing script for EEG data.
Converts raw EDF files to preprocessed .npz files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_subject, save_preprocessed
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess EEG data")
    parser.add_argument('--config', type=str, default='configs/data/physionet.yaml',
                       help='Path to dataset config')
    parser.add_argument('--raw_dir', type=str, required=True,
                       help='Directory with raw EEG files')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--subjects', type=str, default=None,
                       help='Comma-separated list of subjects (e.g., S001,S002)')
    parser.add_argument('--num_subjects', type=int, default=None,
                       help='Number of subjects to process')
    parser.add_argument('--input_channels', type=str,
                       default='Fz,Cz,Pz,Oz',
                       help='Comma-separated input channel names')
    parser.add_argument('--target_channels', type=str,
                       default='Fp1,Fp2,F3,F4,C3,C4,P3,P4',
                       help='Comma-separated target channel names')
    
    return parser.parse_args()


def find_edf_files(raw_dir: Path, subject_id: str) -> List[Path]:
    """Find all EDF files for a subject."""
    # Try different naming patterns
    patterns = [
        f"*{subject_id}*R*.edf",  # PhysioNet pattern
        f"{subject_id}/*.edf",
        f"S{subject_id}/**/*.edf",
    ]
    
    files = []
    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))
    
    return sorted(set(files))


def preprocess_physionet(
    raw_dir: str,
    output_dir: str,
    subjects: List[str],
    input_channels: List[str],
    target_channels: List[str],
    config: dict,
) -> dict:
    """
    Preprocess PhysioNet dataset.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'processed': 0,
        'failed': 0,
        'total_segments': 0,
        'valid_segments': 0,
    }
    
    for subject_id in tqdm(subjects, desc="Processing subjects"):
        try:
            # Find EDF files
            edf_files = find_edf_files(raw_dir, subject_id)
            
            if not edf_files:
                print(f"Warning: No EDF files found for {subject_id}")
                stats['failed'] += 1
                continue
            
            # Process each run and combine
            all_input_segments = []
            all_target_segments = []
            all_valid_masks = []
            
            for edf_path in edf_files:
                try:
                    result = preprocess_subject(
                        str(edf_path),
                        input_channels,
                        target_channels,
                        config
                    )
                    
                    all_input_segments.append(result['input_eeg'])
                    all_target_segments.append(result['target_eeg'])
                    all_valid_masks.append(result['valid_mask'])
                    
                except Exception as e:
                    print(f"Warning: Failed to process {edf_path}: {e}")
                    continue
            
            if not all_input_segments:
                stats['failed'] += 1
                continue
            
            # Concatenate all runs
            import numpy as np
            combined = {
                'input_eeg': np.concatenate(all_input_segments, axis=0),
                'target_eeg': np.concatenate(all_target_segments, axis=0),
                'valid_mask': np.concatenate(all_valid_masks, axis=0),
            }
            
            # Save
            output_path = output_dir / f"{subject_id}.npz"
            save_preprocessed(combined, str(output_path))
            
            stats['processed'] += 1
            stats['total_segments'] += int(len(combined['valid_mask']))
            stats['valid_segments'] += int(combined['valid_mask'].sum())
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            stats['failed'] += 1
    
    return stats


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Parse channels
    input_channels = [ch.strip() for ch in args.input_channels.split(',')]
    target_channels = [ch.strip() for ch in args.target_channels.split(',')]
    
    # Get subjects
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(',')]
    else:
        # Generate subject IDs
        num = args.num_subjects or config.dataset.get('num_subjects', 5)
        subjects = [f"S{i:03d}" for i in range(1, num + 1)]
    
    print(f"Preprocessing {len(subjects)} subjects")
    print(f"Input channels: {input_channels}")
    print(f"Target channels: {target_channels}")
    print(f"Output directory: {args.output_dir}")
    
    # Preprocessing config
    preprocess_config = {
        'target_fs': config.get('target_sample_rate', 200),
        'segment_length_sec': config.preprocessing.get('segment_length_sec', 10.0),
        'segment_overlap_sec': config.preprocessing.get('segment_overlap_sec', 5.0),
        'filter_low': config.preprocessing.get('filter_low', 0.5),
        'filter_high': config.preprocessing.get('filter_high', 45.0),
        'artifact_threshold': config.preprocessing.get('artifact_threshold', 100.0),
    }
    
    # Run preprocessing
    stats = preprocess_physionet(
        args.raw_dir,
        args.output_dir,
        subjects,
        input_channels,
        target_channels,
        preprocess_config,
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Processed: {stats['processed']} subjects")
    print(f"Failed: {stats['failed']} subjects")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Valid segments: {stats['valid_segments']}")
    print(f"Rejection rate: {(1 - stats['valid_segments']/max(1, stats['total_segments']))*100:.1f}%")
    
    # Save processing stats
    stats_path = Path(args.output_dir) / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")


if __name__ == '__main__':
    main()
