"""
Temporal Decoder.
Decodes REVE-style patch embeddings back to time-domain EEG signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecoder(nn.Module):
    """
    Decodes patch embeddings to time-domain EEG signals.
    
    Architecture:
    1. Merge patches using transposed convolutions
    2. Upsample to full time resolution
    3. Final projection to 1 dimension per channel
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_patches: int = 11,
        patch_size: int = 200,
        patch_stride: int = 180,
        output_length: int = 2000,
        num_target_channels: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of input embeddings
            num_patches: Number of patches from REVE
            patch_size: Size of each patch in samples
            patch_stride: Stride between patches
            output_length: Target output length in samples
            num_target_channels: Number of output channels
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.output_length = output_length
        self.num_target_channels = num_target_channels
        
        # Project embeddings to patch reconstructions
        self.patch_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, patch_size),
        )
        
        # Overlap-add reconstruction with learnable blending
        self.blend_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=21, padding=10),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=21, padding=10),
        )
        
        # Refinement after overlap-add
        self.refinement = nn.Sequential(
            nn.Conv1d(num_target_channels, num_target_channels * 4, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_target_channels * 4, num_target_channels * 2, kernel_size=15, padding=7),
            nn.GELU(),
            nn.Conv1d(num_target_channels * 2, num_target_channels, kernel_size=15, padding=7),
        )
        
        # Final output projection
        self.output_proj = nn.Conv1d(num_target_channels, num_target_channels, kernel_size=1)
    
    def forward(
        self,
        expanded_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode embeddings to time-domain EEG.
        
        Args:
            expanded_embeddings: (batch, num_patches, num_channels, hidden_dim)
                                e.g., (B, 11, 8, 512)
        
        Returns:
            reconstructed: (batch, num_channels, output_length)
                          e.g., (B, 8, 2000)
        """
        B, P, C, H = expanded_embeddings.shape
        
        # Decode each patch
        # (B, P, C, H) -> (B, P, C, patch_size)
        decoded_patches = self.patch_decoder(expanded_embeddings)
        
        # Overlap-add reconstruction
        reconstructed = self._overlap_add(decoded_patches)  # (B, C, output_length)
        
        # Refinement
        reconstructed = reconstructed + self.refinement(reconstructed)
        
        # Final projection
        output = self.output_proj(reconstructed)
        
        return output
    
    def _overlap_add(
        self,
        patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct signal using overlap-add.
        
        Args:
            patches: (batch, num_patches, num_channels, patch_size)
        
        Returns:
            signal: (batch, num_channels, output_length)
        """
        B, P, C, T = patches.shape
        
        # Initialize output and weight buffers
        output = torch.zeros(B, C, self.output_length, device=patches.device)
        weight = torch.zeros(B, C, self.output_length, device=patches.device)
        
        # Create triangular window for overlap-add
        window = torch.ones(T, device=patches.device)
        # Fade in/out at edges
        fade_len = min(T // 4, 50)
        window[:fade_len] = torch.linspace(0, 1, fade_len, device=patches.device)
        window[-fade_len:] = torch.linspace(1, 0, fade_len, device=patches.device)
        
        # Add each patch
        for i in range(P):
            start = i * self.patch_stride
            end = start + self.patch_size
            
            if end <= self.output_length:
                patch = patches[:, i]  # (B, C, T)
                output[:, :, start:end] += patch * window
                weight[:, :, start:end] += window
        
        # Handle edge case: if some samples have zero weight
        weight = torch.clamp(weight, min=1e-8)
        
        # Normalize by weight
        output = output / weight
        
        return output


class SimpleDecoder(nn.Module):
    """
    A simpler decoder that directly maps embeddings to output.
    Useful for debugging and baselines.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_patches: int = 11,
        output_length: int = 2000,
        num_target_channels: int = 8,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.output_length = output_length
        self.num_target_channels = num_target_channels
        
        # Simple MLP
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B, P*C*H)
            nn.Linear(num_patches * num_target_channels * hidden_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, num_target_channels * output_length),
        )
    
    def forward(self, expanded_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expanded_embeddings: (B, P, C, H)
        Returns:
            output: (B, C, T)
        """
        B = expanded_embeddings.shape[0]
        output = self.decoder(expanded_embeddings)
        return output.view(B, self.num_target_channels, self.output_length)
