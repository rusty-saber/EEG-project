"""
Electrode position utilities.
Loads 3D positions from MNE montages and REVE position bank.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# Standard 10-20/10-10 electrode positions (normalized to unit sphere)
# Coordinates: (x, y, z) where x=right, y=front, z=up
STANDARD_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    # Midline electrodes (input)
    'Fz': (0.0, 0.38, 0.92),
    'Cz': (0.0, 0.0, 1.0),
    'Pz': (0.0, -0.38, 0.92),
    'Oz': (0.0, -0.71, 0.71),
    
    # Frontal electrodes
    'Fp1': (-0.31, 0.95, 0.0),
    'Fp2': (0.31, 0.95, 0.0),
    'F3': (-0.55, 0.54, 0.64),
    'F4': (0.55, 0.54, 0.64),
    'F7': (-0.81, 0.59, 0.0),
    'F8': (0.81, 0.59, 0.0),
    
    # Central electrodes
    'C3': (-0.71, 0.0, 0.71),
    'C4': (0.71, 0.0, 0.71),
    
    # Parietal electrodes
    'P3': (-0.55, -0.54, 0.64),
    'P4': (0.55, -0.54, 0.64),
    'P7': (-0.81, -0.59, 0.0),
    'P8': (0.81, -0.59, 0.0),
    
    # Temporal electrodes
    'T7': (-1.0, 0.0, 0.0),
    'T8': (1.0, 0.0, 0.0),
    
    # Occipital electrodes
    'O1': (-0.31, -0.95, 0.0),
    'O2': (0.31, -0.95, 0.0),
}


def get_electrode_positions(
    channel_names: List[str],
    source: str = 'standard',
    as_tensor: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get 3D positions for a list of electrode names.
    
    Args:
        channel_names: List of electrode names (e.g., ['Fz', 'Cz', 'Pz', 'Oz'])
        source: Position source - 'standard' for built-in, 'mne' for MNE montage,
                'reve' for REVE position bank
        as_tensor: If True, return torch.Tensor, else numpy array
        
    Returns:
        Positions array of shape (num_channels, 3)
    """
    if source == 'standard':
        positions = _get_standard_positions(channel_names)
    elif source == 'mne':
        positions = _get_mne_positions(channel_names)
    elif source == 'reve':
        positions = _get_reve_positions(channel_names)
    else:
        raise ValueError(f"Unknown position source: {source}")
    
    if as_tensor:
        return torch.tensor(positions, dtype=torch.float32)
    return positions


def _get_standard_positions(channel_names: List[str]) -> np.ndarray:
    """Get positions from built-in standard dictionary."""
    positions = []
    for name in channel_names:
        # Handle case variations
        name_upper = name.upper()
        name_cap = name.capitalize()
        
        if name in STANDARD_POSITIONS:
            positions.append(STANDARD_POSITIONS[name])
        elif name_upper in STANDARD_POSITIONS:
            positions.append(STANDARD_POSITIONS[name_upper])
        elif name_cap in STANDARD_POSITIONS:
            positions.append(STANDARD_POSITIONS[name_cap])
        else:
            raise ValueError(f"Unknown electrode: {name}")
    
    return np.array(positions, dtype=np.float32)


def _get_mne_positions(channel_names: List[str]) -> np.ndarray:
    """Get positions from MNE standard_1020 montage."""
    try:
        import mne
    except ImportError:
        raise ImportError("MNE is required for MNE montage positions")
    
    # Load standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    
    positions = []
    for name in channel_names:
        # Try exact match first
        if name in pos_dict:
            positions.append(pos_dict[name])
        # Try uppercase
        elif name.upper() in pos_dict:
            positions.append(pos_dict[name.upper()])
        else:
            raise ValueError(f"Electrode {name} not found in MNE montage")
    
    return np.array(positions, dtype=np.float32)


def _get_reve_positions(channel_names: List[str]) -> np.ndarray:
    """Get positions from REVE position bank."""
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError("transformers is required for REVE position bank")
    
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions",
        trust_remote_code=True
    )
    
    positions = pos_bank(channel_names)
    return positions.numpy()


def get_input_target_positions(
    input_channels: List[str],
    target_channels: List[str],
    source: str = 'standard'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get positions for both input and target channels.
    
    Returns:
        input_positions: (num_input, 3)
        target_positions: (num_target, 3)
    """
    input_pos = get_electrode_positions(input_channels, source=source)
    target_pos = get_electrode_positions(target_channels, source=source)
    return input_pos, target_pos


def add_position_jitter(
    positions: torch.Tensor,
    jitter_std: float = 0.01
) -> torch.Tensor:
    """
    Add Gaussian jitter to electrode positions for augmentation.
    
    Args:
        positions: (num_channels, 3) or (batch, num_channels, 3)
        jitter_std: Standard deviation of jitter (in same units as positions)
        
    Returns:
        Jittered positions with same shape
    """
    noise = torch.randn_like(positions) * jitter_std
    return positions + noise


def save_positions_json(
    input_channels: List[str],
    target_channels: List[str],
    output_path: str,
    source: str = 'standard'
) -> None:
    """Save electrode positions to JSON file."""
    import json
    
    input_pos = get_electrode_positions(input_channels, source=source, as_tensor=False)
    target_pos = get_electrode_positions(target_channels, source=source, as_tensor=False)
    
    data = {
        'input_channels': {
            name: pos.tolist() for name, pos in zip(input_channels, input_pos)
        },
        'target_channels': {
            name: pos.tolist() for name, pos in zip(target_channels, target_pos)
        },
        'source': source
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
