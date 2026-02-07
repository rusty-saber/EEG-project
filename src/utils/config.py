"""
Configuration loading and management.
Supports YAML config files with inheritance and CLI overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import OmegaConf, DictConfig


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(
    config_paths: Union[str, Path, List[Union[str, Path]]],
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Load and merge configuration files.
    
    Args:
        config_paths: Single path or list of paths to YAML config files.
                     Later configs override earlier ones.
        overrides: Dictionary of overrides to apply on top.
        
    Returns:
        Merged OmegaConf DictConfig.
        
    Example:
        config = load_config(['configs/base.yaml', 'configs/stage1.yaml'])
        config = load_config('configs/stage1.yaml', overrides={'training.epochs': 10})
    """
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    
    # Load and merge configs in order
    merged = OmegaConf.create()
    for path in config_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(merged, cfg)
    
    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        merged = OmegaConf.merge(merged, override_cfg)
    
    # Resolve interpolations
    OmegaConf.resolve(merged)
    
    return merged


def save_config(config: DictConfig, path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        OmegaConf.save(config, f)


def get_experiment_name(config: DictConfig, timestamp: Optional[str] = None) -> str:
    """
    Generate experiment name from config.
    
    Format: {stage}_{dataset}_{channels}_{timestamp}
    Example: stage1_physionet_4to8_20260202_1830
    """
    from datetime import datetime
    
    stage = config.get('stage', 1)
    dataset = config.get('dataset', {}).get('name', 'unknown')
    channels = f"{config.get('num_input_channels', 4)}to{config.get('num_target_channels', 8)}"
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    return f"stage{stage}_{dataset}_{channels}_{timestamp}"


def print_config(config: DictConfig) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(config))
