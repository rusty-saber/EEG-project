"""
EEG Preprocessing pipeline.
Handles filtering, resampling, segmentation, normalization, and artifact rejection.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from tqdm import tqdm


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low_freq: float = 0.5,
    high_freq: float = 45.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.
    
    Args:
        data: EEG data of shape (channels, samples) or (samples,)
        fs: Sampling frequency in Hz
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data with same shape as input
    """
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are in valid range
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, ch) for ch in data])


def resample_data(
    data: np.ndarray,
    original_fs: float,
    target_fs: float
) -> np.ndarray:
    """
    Resample EEG data to target sampling frequency.
    
    Args:
        data: EEG data of shape (channels, samples)
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz
        
    Returns:
        Resampled data
    """
    if original_fs == target_fs:
        return data
    
    num_samples = data.shape[-1]
    target_samples = int(num_samples * target_fs / original_fs)
    
    if data.ndim == 1:
        return signal.resample(data, target_samples)
    else:
        return np.array([signal.resample(ch, target_samples) for ch in data])


def segment_data(
    data: np.ndarray,
    segment_length: int,
    overlap: int = 0
) -> np.ndarray:
    """
    Segment continuous EEG into fixed-length windows.
    
    Args:
        data: EEG data of shape (channels, samples)
        segment_length: Length of each segment in samples
        overlap: Overlap between segments in samples
        
    Returns:
        Segmented data of shape (num_segments, channels, segment_length)
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_channels, n_samples = data.shape
    stride = segment_length - overlap
    
    # Calculate number of segments
    n_segments = max(1, (n_samples - segment_length) // stride + 1)
    
    segments = []
    for i in range(n_segments):
        start = i * stride
        end = start + segment_length
        if end <= n_samples:
            segments.append(data[:, start:end])
    
    if not segments:
        # If data is shorter than segment_length, pad with zeros
        padded = np.zeros((n_channels, segment_length))
        padded[:, :n_samples] = data
        return padded.reshape(1, n_channels, segment_length)
    
    return np.array(segments)


def z_normalize(
    data: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-normalize data per channel.
    
    Args:
        data: EEG data
        axis: Axis along which to compute mean/std
        eps: Small value to prevent division by zero
        
    Returns:
        (normalized_data, means, stds)
    """
    means = np.mean(data, axis=axis, keepdims=True)
    stds = np.std(data, axis=axis, keepdims=True)
    stds = np.maximum(stds, eps)
    
    normalized = (data - means) / stds
    
    return normalized, means.squeeze(), stds.squeeze()


def detect_artifacts(
    data: np.ndarray,
    fs: float,
    amplitude_threshold: float = 100.0,
    flat_threshold: float = 0.1,
    flat_window_sec: float = 1.0
) -> np.ndarray:
    """
    Detect artifacts in EEG segments.
    
    Args:
        data: EEG data of shape (num_segments, channels, samples)
               or (channels, samples)
        fs: Sampling frequency
        amplitude_threshold: Max amplitude in μV (absolute value)
        flat_threshold: Min std for non-flat signal
        flat_window_sec: Window size for flatness check
        
    Returns:
        Boolean mask of shape (num_segments,) - True means artifact-free
    """
    if data.ndim == 2:
        data = data.reshape(1, *data.shape)
    
    n_segments = data.shape[0]
    valid_mask = np.ones(n_segments, dtype=bool)
    
    flat_window = int(flat_window_sec * fs)
    
    for i in range(n_segments):
        segment = data[i]
        
        # Check amplitude threshold
        if np.any(np.abs(segment) > amplitude_threshold):
            valid_mask[i] = False
            continue
        
        # Check for flat segments
        for ch in segment:
            for start in range(0, len(ch) - flat_window, flat_window):
                window = ch[start:start + flat_window]
                if np.std(window) < flat_threshold:
                    valid_mask[i] = False
                    break
            if not valid_mask[i]:
                break
    
    return valid_mask


def preprocess_eeg(
    data: np.ndarray,
    original_fs: float,
    target_fs: float = 200.0,
    segment_length_sec: float = 10.0,
    segment_overlap_sec: float = 5.0,
    filter_low: float = 0.5,
    filter_high: float = 45.0,
    artifact_threshold: float = 100.0,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline for EEG data.
    
    Args:
        data: Raw EEG data of shape (channels, samples)
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency
        segment_length_sec: Segment length in seconds
        segment_overlap_sec: Segment overlap in seconds
        filter_low: Low cutoff for bandpass filter
        filter_high: High cutoff for bandpass filter
        artifact_threshold: Amplitude threshold for artifact rejection
        normalize: Whether to z-normalize
        
    Returns:
        Dictionary containing:
        - 'segments': (num_valid_segments, channels, segment_length)
        - 'valid_mask': Boolean mask of artifact-free segments
        - 'means': Mean per channel (if normalized)
        - 'stds': Std per channel (if normalized)
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(data, original_fs, filter_low, filter_high)
    
    # Step 2: Resample
    resampled = resample_data(filtered, original_fs, target_fs)
    
    # Step 3: Segment
    segment_length = int(segment_length_sec * target_fs)
    segment_overlap = int(segment_overlap_sec * target_fs)
    segments = segment_data(resampled, segment_length, segment_overlap)
    
    # Step 4: Artifact rejection
    valid_mask = detect_artifacts(
        segments, target_fs, 
        amplitude_threshold=artifact_threshold
    )
    
    # Step 5: Normalize
    if normalize:
        # Normalize each segment independently
        normalized_segments = []
        all_means = []
        all_stds = []
        
        for seg in segments:
            norm_seg, means, stds = z_normalize(seg)
            normalized_segments.append(norm_seg)
            all_means.append(means)
            all_stds.append(stds)
        
        segments = np.array(normalized_segments)
        means = np.array(all_means)
        stds = np.array(all_stds)
    else:
        means = None
        stds = None
    
    return {
        'segments': segments.astype(np.float32),
        'valid_mask': valid_mask,
        'means': means,
        'stds': stds,
    }


def preprocess_subject(
    edf_path: str,
    input_channels: List[str],
    target_channels: List[str],
    config: dict
) -> Dict[str, np.ndarray]:
    """
    Preprocess a single subject's EDF file.
    
    Args:
        edf_path: Path to EDF file
        input_channels: List of input channel names
        target_channels: List of target channel names
        config: Preprocessing configuration
        
    Returns:
        Dictionary with 'input_eeg', 'target_eeg', 'valid_mask'
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE is required for reading EDF files")
    
    # Load raw data
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Get sampling frequency
    original_fs = raw.info['sfreq']
    
    # Select channels
    all_channels = input_channels + target_channels
    
    # Handle channel name variations (PhysioNet uses dot-padded names like 'Fz..', 'Cz..', 'Fp1.')
    available_channels = raw.ch_names
    
    # Create a mapping from standard names to available channel names
    # PhysioNet pads channel names with dots to 4 characters (e.g., Fz -> Fz.., C3 -> C3.., Fp1 -> Fp1.)
    def normalize_channel_name(name):
        """Remove trailing dots and convert to title case."""
        return name.rstrip('.').capitalize()
    
    # Build a lookup dictionary for available channels
    available_normalized = {}
    for ch in available_channels:
        norm = normalize_channel_name(ch)
        available_normalized[norm.lower()] = ch  # Use lowercase for matching
    
    channel_mapping = {}
    for ch in all_channels:
        ch_lower = ch.lower()
        if ch_lower in available_normalized:
            channel_mapping[ch] = available_normalized[ch_lower]
        # Also try with case variations
        elif ch.capitalize().lower() in available_normalized:
            channel_mapping[ch] = available_normalized[ch.capitalize().lower()]
    
    if len(channel_mapping) < len(all_channels):
        missing = set(all_channels) - set(channel_mapping.keys())
        # Print available channels to help debug
        raise ValueError(f"Missing channels: {missing}. Available: {[normalize_channel_name(c) for c in available_channels[:20]]}")
    
    # Pick channels in the correct order
    ordered_channels = [channel_mapping[ch] for ch in all_channels]
    raw.pick(ordered_channels)
    
    # Get data
    data = raw.get_data()  # (channels, samples)
    
    # Convert to μV if needed (MNE uses V by default)
    data = data * 1e6
    
    # Split into input and target
    n_input = len(input_channels)
    input_data = data[:n_input]
    target_data = data[n_input:]
    
    # Preprocess input
    input_result = preprocess_eeg(
        input_data,
        original_fs,
        target_fs=config.get('target_fs', 200.0),
        segment_length_sec=config.get('segment_length_sec', 10.0),
        segment_overlap_sec=config.get('segment_overlap_sec', 5.0),
        filter_low=config.get('filter_low', 0.5),
        filter_high=config.get('filter_high', 45.0),
        artifact_threshold=config.get('artifact_threshold', 100.0),
    )
    
    # Preprocess target with same parameters
    target_result = preprocess_eeg(
        target_data,
        original_fs,
        target_fs=config.get('target_fs', 200.0),
        segment_length_sec=config.get('segment_length_sec', 10.0),
        segment_overlap_sec=config.get('segment_overlap_sec', 5.0),
        filter_low=config.get('filter_low', 0.5),
        filter_high=config.get('filter_high', 45.0),
        artifact_threshold=config.get('artifact_threshold', 100.0),
    )
    
    # Combine valid masks (both input and target must be artifact-free)
    valid_mask = input_result['valid_mask'] & target_result['valid_mask']
    
    return {
        'input_eeg': input_result['segments'],
        'target_eeg': target_result['segments'],
        'valid_mask': valid_mask,
        'input_means': input_result['means'],
        'input_stds': input_result['stds'],
        'target_means': target_result['means'],
        'target_stds': target_result['stds'],
    }


def save_preprocessed(
    data: Dict[str, np.ndarray],
    output_path: str
) -> None:
    """Save preprocessed data to .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        input_eeg=data['input_eeg'],
        target_eeg=data['target_eeg'],
        valid_mask=data['valid_mask']
    )


def load_preprocessed(path: str) -> Dict[str, np.ndarray]:
    """Load preprocessed data from .npz file."""
    data = np.load(path)
    return {
        'input_eeg': data['input_eeg'],
        'target_eeg': data['target_eeg'],
        'valid_mask': data['valid_mask'],
    }
