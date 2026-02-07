"""
PyTorch Dataset for EEG channel expansion.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EEGChannelExpansionDataset(Dataset):
    """
    Dataset for 4→8 EEG channel expansion.
    
    Loads preprocessed .npz files and returns input/target EEG pairs
    with electrode positions.
    
    Expected .npz structure:
        - input_eeg: (num_segments, 4, 2000)
        - target_eeg: (num_segments, 8, 2000)
        - valid_mask: (num_segments,) boolean
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        subject_ids: List[str],
        input_channels: List[str],
        target_channels: List[str],
        input_positions: Optional[torch.Tensor] = None,
        target_positions: Optional[torch.Tensor] = None,
        position_source: str = 'standard',
        augment: bool = False,
        position_jitter_std: float = 0.01,
        use_all_segments: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing preprocessed .npz files
            subject_ids: List of subject IDs to include (e.g., ['S001', 'S002'])
            input_channels: List of input channel names
            target_channels: List of target channel names
            input_positions: Pre-computed input positions (num_input, 3)
            target_positions: Pre-computed target positions (num_target, 3)
            position_source: Source for positions if not provided
            augment: Whether to apply augmentations
            position_jitter_std: Std for position jitter augmentation
            use_all_segments: If True, ignore valid_mask and use all segments
        """
        self.data_dir = Path(data_dir)
        self.subject_ids = subject_ids
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.augment = augment
        self.position_jitter_std = position_jitter_std
        self.use_all_segments = use_all_segments
        
        # Load positions
        if input_positions is not None:
            self.input_positions = input_positions
        else:
            from ..utils.positions import get_electrode_positions
            self.input_positions = get_electrode_positions(
                input_channels, source=position_source
            )
        
        if target_positions is not None:
            self.target_positions = target_positions
        else:
            from ..utils.positions import get_electrode_positions
            self.target_positions = get_electrode_positions(
                target_channels, source=position_source
            )
        
        # Load all data and build index
        self.samples = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load all subject data and build sample index."""
        for subject_id in self.subject_ids:
            npz_path = self.data_dir / f"{subject_id}.npz"
            
            if not npz_path.exists():
                print(f"Warning: {npz_path} not found, skipping")
                continue
            
            data = np.load(npz_path)
            input_eeg = data['input_eeg']
            target_eeg = data['target_eeg']
            valid_mask = data['valid_mask']
            
            # Use all segments or only valid ones
            if self.use_all_segments:
                valid_indices = np.arange(len(input_eeg))
            else:
                valid_indices = np.where(valid_mask)[0]
            
            for idx in valid_indices:
                self.samples.append({
                    'subject_id': subject_id,
                    'segment_idx': int(idx),
                    'input_eeg': input_eeg[idx],
                    'target_eeg': target_eeg[idx],
                })
        
        print(f"Loaded {len(self.samples)} samples from {len(self.subject_ids)} subjects")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - input_eeg: (4, 2000) input EEG
            - target_eeg: (8, 2000) target EEG
            - input_positions: (4, 3) electrode positions
            - target_positions: (8, 3) electrode positions
            - subject_id: Subject identifier
            - segment_idx: Segment index within subject
        """
        sample = self.samples[idx]
        
        input_eeg = torch.tensor(sample['input_eeg'], dtype=torch.float32)
        target_eeg = torch.tensor(sample['target_eeg'], dtype=torch.float32)
        
        # Clone positions (so augmentation doesn't modify originals)
        input_positions = self.input_positions.clone()
        target_positions = self.target_positions.clone()
        
        # Apply augmentations
        if self.augment:
            input_eeg, target_eeg, input_positions, target_positions = self._augment(
                input_eeg, target_eeg, input_positions, target_positions
            )
        
        return {
            'input_eeg': input_eeg,
            'target_eeg': target_eeg,
            'input_positions': input_positions,
            'target_positions': target_positions,
            'subject_id': sample['subject_id'],
            'segment_idx': sample['segment_idx'],
        }
    
    def _augment(
        self,
        input_eeg: torch.Tensor,
        target_eeg: torch.Tensor,
        input_positions: torch.Tensor,
        target_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply augmentations."""
        # Position jitter
        if self.position_jitter_std > 0:
            from ..utils.positions import add_position_jitter
            input_positions = add_position_jitter(
                input_positions, self.position_jitter_std
            )
            target_positions = add_position_jitter(
                target_positions, self.position_jitter_std
            )
        
        # TODO: Add more augmentations (noise, time shift, etc.)
        
        return input_eeg, target_eeg, input_positions, target_positions


def create_dataloaders(
    data_dir: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: Optional[List[str]] = None,
    input_channels: List[str] = ['Fz', 'Cz', 'Pz', 'Oz'],
    target_channels: List[str] = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    augment_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        data_dir: Directory containing preprocessed data
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
        test_subjects: Optional list of test subject IDs
        input_channels: Input channel names
        target_channels: Target channel names
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        augment_train: Whether to augment training data
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataLoaders
    """
    from ..utils.positions import get_input_target_positions
    
    # Get positions once (shared across datasets)
    input_pos, target_pos = get_input_target_positions(
        input_channels, target_channels
    )
    
    # Create datasets
    train_dataset = EEGChannelExpansionDataset(
        data_dir=data_dir,
        subject_ids=train_subjects,
        input_channels=input_channels,
        target_channels=target_channels,
        input_positions=input_pos,
        target_positions=target_pos,
        augment=augment_train,
    )
    
    val_dataset = EEGChannelExpansionDataset(
        data_dir=data_dir,
        subject_ids=val_subjects,
        input_channels=input_channels,
        target_channels=target_channels,
        input_positions=input_pos,
        target_positions=target_pos,
        augment=False,
    )
    
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    
    if test_subjects:
        test_dataset = EEGChannelExpansionDataset(
            data_dir=data_dir,
            subject_ids=test_subjects,
            input_channels=input_channels,
            target_channels=target_channels,
            input_positions=input_pos,
            target_positions=target_pos,
            augment=False,
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return loaders


def get_subject_ids(
    split: str,
    dataset: str = 'physionet',
    splits_path: Optional[str] = None
) -> List[str]:
    """
    Get subject IDs for a given split.
    
    Args:
        split: 'train', 'val', or 'test'
        dataset: Dataset name
        splits_path: Path to subject_splits.json
        
    Returns:
        List of subject IDs
    """
    if splits_path and Path(splits_path).exists():
        with open(splits_path) as f:
            splits = json.load(f)
        return splits[dataset][split]
    
    # Default PhysioNet splits
    if dataset == 'physionet':
        if split == 'train':
            return [f"S{i:03d}" for i in range(1, 81)]
        elif split == 'val':
            return [f"S{i:03d}" for i in range(81, 95)]
        elif split == 'test':
            return [f"S{i:03d}" for i in range(95, 110)]
    
    raise ValueError(f"Unknown split: {split} for dataset: {dataset}")
