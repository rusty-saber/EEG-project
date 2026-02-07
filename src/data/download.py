"""
Data download utilities for PhysioNet and OpenBMI datasets.
"""

import os
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def download_physionet(
    output_dir: str,
    subjects: Optional[List[int]] = None,
    num_subjects: int = 5,
    runs: Optional[List[int]] = None
) -> Path:
    """
    Download PhysioNet EEG Motor Movement/Imagery Dataset.
    
    Uses MNE's built-in download functionality.
    
    Args:
        output_dir: Directory to save data
        subjects: Specific subject IDs to download (1-109)
        num_subjects: Number of subjects if subjects not specified
        runs: Run numbers to download (1-14), default all
        
    Returns:
        Path to downloaded data directory
    """
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError:
        raise ImportError("MNE is required for downloading PhysioNet data")
    
    output_dir = Path(output_dir) / 'physionet'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default subjects
    if subjects is None:
        subjects = list(range(1, num_subjects + 1))
    
    # Default runs (use a subset for faster download: rest, motor execution, motor imagery)
    if runs is None:
        runs = [1, 2, 3, 4, 5, 6]  # Run 1=baseline, 2=baseline, 3-4=motor, 5-6=imagery
    
    print(f"Downloading PhysioNet data for {len(subjects)} subjects...")
    print(f"Runs: {runs}")
    
    for subject_id in tqdm(subjects, desc="Downloading subjects"):
        try:
            # Download the runs for this subject
            # MNE eegbci.load_data expects subject as positional arg
            raw_fnames = eegbci.load_data(
                subject_id,  # positional argument
                runs,
                path=str(output_dir),
                update_path=False
            )
            print(f"  Downloaded subject {subject_id}: {len(raw_fnames)} files")
        except Exception as e:
            print(f"Warning: Failed to download subject {subject_id}: {e}")
            continue
    
    print(f"Downloaded data to {output_dir}")
    return output_dir


def download_openbmi(
    output_dir: str,
    subjects: Optional[List[int]] = None,
    num_subjects: int = 5
) -> Path:
    """
    Download OpenBMI dataset.
    
    Note: OpenBMI requires manual download from GigaDB or a mirror.
    This function provides instructions and checks for existing data.
    
    Args:
        output_dir: Directory to save data
        subjects: Specific subject IDs to download (1-54)
        num_subjects: Number of subjects if subjects not specified
        
    Returns:
        Path to data directory
    """
    output_dir = Path(output_dir) / 'openbmi'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    existing_files = list(output_dir.glob('*.mat'))
    if existing_files:
        print(f"Found {len(existing_files)} existing OpenBMI files")
        return output_dir
    
    print("=" * 60)
    print("OpenBMI Dataset Manual Download Required")
    print("=" * 60)
    print("""
The OpenBMI dataset must be downloaded manually:

1. Go to: https://gigadb.org/dataset/100542
2. Download the MI (Motor Imagery) session files
3. Extract to: {output_dir}

Expected file format: sess01_subj{{XX}}_EEG_MI.mat

Alternative: Use MOABB library for automatic download:
    pip install moabb
    
Then use:
    from moabb.datasets import Lee2019_MI
    dataset = Lee2019_MI()
    dataset.download()
""".format(output_dir=output_dir))
    
    return output_dir


def verify_download(data_dir: str, dataset: str, expected_subjects: int) -> bool:
    """
    Verify that download completed successfully.
    
    Args:
        data_dir: Data directory
        dataset: 'physionet' or 'openbmi'
        expected_subjects: Expected number of subjects
        
    Returns:
        True if verification passes
    """
    data_dir = Path(data_dir) / dataset
    
    if dataset == 'physionet':
        # Check for EDF files (PhysioNet stores as files, not directories)
        edf_files = list(data_dir.glob('**/*.edf'))
        # Each subject has multiple runs
        actual = len(set(f.parent.name for f in edf_files)) if edf_files else 0
        if actual == 0:
            # Also try counting based on file naming
            actual = len(set(str(f.name).split('R')[0] for f in edf_files if 'R' in f.name))
    elif dataset == 'openbmi':
        # Check for .mat files
        mat_files = list(data_dir.glob('*.mat'))
        actual = len(mat_files)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if actual >= expected_subjects:
        print(f"[OK] Verified: Found {actual} subjects (expected {expected_subjects})")
        return True
    else:
        print(f"[INCOMPLETE] Found {actual} subjects (expected {expected_subjects})")
        return False
