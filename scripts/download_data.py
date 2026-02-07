#!/usr/bin/env python
"""
Download EEG data script.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.download import download_physionet, download_openbmi, verify_download
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Download EEG datasets")
    parser.add_argument('--dataset', type=str, default='physionet',
                       choices=['physionet', 'openbmi'],
                       help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='./data/raw',
                       help='Output directory for downloaded data')
    parser.add_argument('--num_subjects', type=int, default=5,
                       help='Number of subjects to download')
    parser.add_argument('--subjects', type=str, default=None,
                       help='Comma-separated specific subject IDs')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to dataset config file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Downloading {args.dataset} dataset")
    print(f"Output directory: {args.output_dir}")
    
    # Parse subjects if provided
    subjects = None
    if args.subjects:
        subjects = [int(s.strip().replace('S', '')) for s in args.subjects.split(',')]
        print(f"Subjects: {subjects}")
    else:
        print(f"Number of subjects: {args.num_subjects}")
    
    # Download
    if args.dataset == 'physionet':
        output_path = download_physionet(
            output_dir=args.output_dir,
            subjects=subjects,
            num_subjects=args.num_subjects,
        )
    elif args.dataset == 'openbmi':
        output_path = download_openbmi(
            output_dir=args.output_dir,
            subjects=subjects,
            num_subjects=args.num_subjects,
        )
    
    # Verify
    expected = len(subjects) if subjects else args.num_subjects
    verify_download(args.output_dir, args.dataset, expected)
    
    print(f"\nDownload complete: {output_path}")


if __name__ == '__main__':
    main()
