"""
REVE encoder wrapper for feature extraction.
Wraps the pretrained REVE model from HuggingFace.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class REVEWrapper(nn.Module):
    """
    Wrapper around pretrained REVE model for feature extraction.
    
    REVE processes EEG by:
    1. Patching: Splits each channel into 1-second windows (200 samples @ 200Hz)
    2. 4D Positional Encoding: Encodes (x, y, z, time) for each patch
    3. Transformer: Processes all patches together
    
    For 10s input (2000 samples) with 0.1s overlap:
    - patch_size = 200 samples
    - patch_stride = 180 samples  
    - num_patches = (2000 - 200) / 180 + 1 = 11 patches
    """
    
    def __init__(
        self,
        model_name: str = "brain-bzh/reve-base",
        position_bank_name: str = "brain-bzh/position-encoder",
        hidden_dim: int = 512,
        patch_size: int = 200,
        patch_stride: int = 180,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        """
        Initialize REVE wrapper.
        
        Args:
            model_name: HuggingFace model name
            position_bank_name: HuggingFace position bank name
            hidden_dim: REVE hidden dimension
            patch_size: Patch size in samples (1s @ 200Hz)
            patch_stride: Stride between patches
            cache_dir: Directory to cache downloaded model
            trust_remote_code: Trust remote code for model loading
        """
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # Load REVE model
        try:
            from transformers import AutoModel
            
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )
            
            # Get actual hidden dim from config if available
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'd_model'):
                self.hidden_dim = self.model.config.d_model
            
            self._model_loaded = True
            print(f"REVE encoder loaded from {model_name}")
            
        except Exception as e:
            print(f"Warning: Could not load REVE model: {e}")
            print("Using placeholder for development")
            self._model_loaded = False
            
            # Create placeholder for development
            self._placeholder = nn.Linear(patch_size, hidden_dim)
    
    def get_num_patches(self, seq_length: int) -> int:
        """Calculate number of patches for a given sequence length."""
        return max(1, (seq_length - self.patch_size) // self.patch_stride + 1)
    
    def forward(
        self,
        eeg: torch.Tensor,
        positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from EEG using REVE.
        
        Args:
            eeg: (batch_size, num_channels, num_samples)
                 e.g., (16, 4, 2000) for 4 channels, 10s @ 200Hz
            positions: (batch_size, num_channels, 3)
                      3D electrode positions
        
        Returns:
            Dict containing:
            - embeddings: (batch_size, num_patches, num_channels, hidden_dim)
                         e.g., (16, 11, 4, 512)
            - patch_embeddings: (batch_size, num_patches * num_channels, hidden_dim)
                               Flattened version for transformer processing
        """
        batch_size, num_channels, num_samples = eeg.shape
        num_patches = self.get_num_patches(num_samples)
        
        if self._model_loaded:
            # Use actual REVE model
            # REVE expects: eeg (B, C, T), positions (B, C, 3)
            output = self.model(eeg, positions)
            
            # REVE output is (B, C, num_patches, hidden_dim)
            # We need (B, num_patches, C, hidden_dim) for our expansion module
            if hasattr(output, 'last_hidden_state'):
                hidden = output.last_hidden_state
            else:
                hidden = output
            
            # Transpose from (B, C, P, H) to (B, P, C, H)
            if hidden.dim() == 4:
                embeddings = hidden.permute(0, 2, 1, 3)  # (B, P, C, H)
            else:
                # Handle unexpected shapes
                embeddings = hidden
            
            # Get actual hidden dim from output
            actual_hidden_dim = embeddings.shape[-1]
            
            patch_embeddings = embeddings.reshape(
                batch_size, num_patches * num_channels, actual_hidden_dim
            )
            
        else:
            # Use placeholder for development
            # Create mock patches
            patches = []
            for i in range(num_patches):
                start = i * self.patch_stride
                end = start + self.patch_size
                patch = eeg[:, :, start:end]  # (B, C, patch_size)
                patches.append(patch)
            
            patches = torch.stack(patches, dim=1)  # (B, num_patches, C, patch_size)
            
            # Simple projection as placeholder
            embeddings = self._placeholder(patches)  # (B, num_patches, C, hidden_dim)
            
            patch_embeddings = embeddings.reshape(
                batch_size, num_patches * num_channels, self.hidden_dim
            )
        
        return {
            'embeddings': embeddings,
            'patch_embeddings': patch_embeddings,
            'num_patches': num_patches,
            'num_channels': num_channels,
        }
    
    def get_positions_from_names(
        self,
        channel_names: list,
        batch_size: int = 1,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Get positions for channel names using REVE position bank.
        
        Args:
            channel_names: List of electrode names
            batch_size: Batch size to expand positions to
            device: Device to put tensor on
            
        Returns:
            positions: (batch_size, num_channels, 3)
        """
        if self._model_loaded:
            positions = self.position_bank(channel_names)
            positions = positions.expand(batch_size, -1, -1)
            return positions.to(device)
        else:
            # Placeholder
            num_channels = len(channel_names)
            return torch.randn(batch_size, num_channels, 3, device=device)
