"""
Full Channel Expansion Model.
Combines REVE encoder, expansion module, and decoder.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .reve_wrapper import REVEWrapper
from .expansion_module import ChannelExpansionModule
from .decoder import TemporalDecoder


class ChannelExpansionModel(nn.Module):
    """
    Complete 4→8 EEG channel reconstruction model.
    
    Pipeline:
    1. REVE Encoder: Extract spatio-temporal features from 4-channel input
    2. Expansion Module: Expand 4-channel features to 8-channel features
    3. Decoder: Reconstruct 8-channel time-domain EEG
    """
    
    def __init__(
        self,
        # REVE settings
        reve_model_name: str = "brain-bzh/reve-base",
        position_bank_name: str = "brain-bzh/reve-positions",
        hidden_dim: int = 512,
        patch_size: int = 200,
        patch_stride: int = 180,
        
        # Expansion settings
        expansion_num_heads: int = 8,
        expansion_num_layers: int = 2,
        
        # Decoder settings
        output_length: int = 2000,
        num_target_channels: int = 8,
        
        # Training settings
        freeze_encoder: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize the full model.
        
        Args:
            reve_model_name: HuggingFace model name for REVE
            position_bank_name: HuggingFace position bank name
            hidden_dim: REVE hidden dimension
            patch_size: REVE patch size in samples
            patch_stride: REVE patch stride
            expansion_num_heads: Number of attention heads in expansion
            expansion_num_layers: Number of refinement layers
            output_length: Output sequence length in samples
            num_target_channels: Number of output channels
            freeze_encoder: Whether to freeze REVE encoder
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_target_channels = num_target_channels
        self.freeze_encoder = freeze_encoder
        
        # Calculate num_patches for the given output length
        self.num_patches = (output_length - patch_size) // patch_stride + 1
        
        # Components
        self.reve_encoder = REVEWrapper(
            model_name=reve_model_name,
            position_bank_name=position_bank_name,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            patch_stride=patch_stride,
        )
        
        self.expansion = ChannelExpansionModule(
            hidden_dim=hidden_dim,
            num_heads=expansion_num_heads,
            num_layers=expansion_num_layers,
            dropout=dropout,
        )
        
        self.decoder = TemporalDecoder(
            hidden_dim=hidden_dim,
            num_patches=self.num_patches,
            patch_size=patch_size,
            patch_stride=patch_stride,
            output_length=output_length,
            num_target_channels=num_target_channels,
            dropout=dropout,
        )
        
        # Apply freeze if requested
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self) -> None:
        """Freeze REVE encoder parameters."""
        for param in self.reve_encoder.parameters():
            param.requires_grad = False
        print("REVE encoder frozen")
    
    def _unfreeze_encoder(self) -> None:
        """Unfreeze REVE encoder parameters."""
        for param in self.reve_encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
        print("REVE encoder unfrozen")
    
    def forward(
        self,
        input_eeg: torch.Tensor,
        input_positions: torch.Tensor,
        target_positions: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.
        
        Args:
            input_eeg: (batch, num_input_channels, num_samples)
                      e.g., (B, 4, 2000)
            input_positions: (batch, num_input_channels, 3)
                            e.g., (B, 4, 3)
            target_positions: (batch, num_target_channels, 3)
                             e.g., (B, 8, 3)
            return_intermediates: Whether to return intermediate representations
        
        Returns:
            Dict containing:
            - output: (batch, num_target_channels, num_samples) reconstructed EEG
            - encoder_embeddings: (optional) REVE embeddings
            - expanded_embeddings: (optional) expanded embeddings
        """
        # 1. Encode with REVE
        encoder_out = self.reve_encoder(input_eeg, input_positions)
        encoder_embeddings = encoder_out['embeddings']  # (B, P, C_in, H)
        
        # 2. Expand channels
        expanded_embeddings = self.expansion(
            encoder_embeddings,
            input_positions,
            target_positions,
        )  # (B, P, C_out, H)
        
        # 3. Decode to time domain
        output = self.decoder(expanded_embeddings)  # (B, C_out, T)
        
        result = {'output': output}
        
        if return_intermediates:
            result['encoder_embeddings'] = encoder_embeddings
            result['expanded_embeddings'] = expanded_embeddings
        
        return result
    
    def get_parameter_groups(
        self,
        encoder_lr: float = 1e-6,
        expansion_lr: float = 1e-5,
        decoder_lr: float = 1e-5,
    ) -> list:
        """
        Get parameter groups with differential learning rates.
        
        Args:
            encoder_lr: Learning rate for REVE encoder
            expansion_lr: Learning rate for expansion module
            decoder_lr: Learning rate for decoder
            
        Returns:
            List of parameter group dicts for optimizer
        """
        param_groups = [
            {
                'params': self.reve_encoder.parameters(),
                'lr': encoder_lr,
                'name': 'encoder',
            },
            {
                'params': self.expansion.parameters(),
                'lr': expansion_lr,
                'name': 'expansion',
            },
            {
                'params': self.decoder.parameters(),
                'lr': decoder_lr,
                'name': 'decoder',
            },
        ]
        return param_groups
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'encoder_total': count(self.reve_encoder),
            'encoder_trainable': count_trainable(self.reve_encoder),
            'expansion_total': count(self.expansion),
            'expansion_trainable': count_trainable(self.expansion),
            'decoder_total': count(self.decoder),
            'decoder_trainable': count_trainable(self.decoder),
            'total': count(self),
            'trainable': count_trainable(self),
        }


def create_model(config) -> ChannelExpansionModel:
    """
    Create model from configuration.
    
    Args:
        config: OmegaConf configuration
        
    Returns:
        Initialized ChannelExpansionModel
    """
    return ChannelExpansionModel(
        reve_model_name=config.reve.model_name,
        position_bank_name=config.reve.position_bank,
        hidden_dim=config.reve.hidden_dim,
        patch_size=config.reve.patch_size,
        patch_stride=config.reve.patch_stride,
        expansion_num_heads=config.get('expansion_num_heads', 8),
        expansion_num_layers=config.get('expansion_num_layers', 2),
        output_length=config.data.segment_length,
        num_target_channels=config.get('num_target_channels', 8),
        freeze_encoder=config.freeze_encoder,
        dropout=config.get('regularization', {}).get('dropout', 0.1),
    )
