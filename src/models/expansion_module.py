"""
Channel Expansion Module.
Learns to expand 4-channel REVE embeddings to 8-channel representations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalCrossAttention(nn.Module):
    """
    Cross-attention that uses electrode positions as queries.
    Target electrode positions attend to input electrode embeddings.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        position_dim: int = 3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Project positions to hidden dim for queries
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        input_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand input channel embeddings to target channels via cross-attention.
        
        Args:
            input_embeddings: (batch, num_patches, num_input_channels, hidden_dim)
                             e.g., (B, 11, 4, 512)
            input_positions: (batch, num_input_channels, 3)
                            e.g., (B, 4, 3)
            target_positions: (batch, num_target_channels, 3)
                             e.g., (B, 8, 3)
        
        Returns:
            expanded: (batch, num_patches, num_target_channels, hidden_dim)
                     e.g., (B, 11, 8, 512)
        """
        B, P, C_in, H = input_embeddings.shape
        C_out = target_positions.shape[1]
        
        # Encode positions
        input_pos_emb = self.position_encoder(input_positions)   # (B, C_in, H)
        target_pos_emb = self.position_encoder(target_positions)  # (B, C_out, H)
        
        # Process each patch
        expanded_patches = []
        
        for p in range(P):
            patch_emb = input_embeddings[:, p, :, :]  # (B, C_in, H)
            
            # Add positional information to input embeddings
            patch_emb = patch_emb + input_pos_emb
            
            # Target positions form queries
            q = self.q_proj(target_pos_emb)  # (B, C_out, H)
            k = self.k_proj(patch_emb)       # (B, C_in, H)
            v = self.v_proj(patch_emb)       # (B, C_in, H)
            
            # Reshape for multi-head attention
            q = q.view(B, C_out, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, C_in, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, C_in, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention
            attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, heads, C_out, C_in)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention
            out = torch.matmul(attn, v)  # (B, heads, C_out, head_dim)
            out = out.transpose(1, 2).contiguous().view(B, C_out, H)
            out = self.out_proj(out)
            
            expanded_patches.append(out)
        
        # Stack patches
        expanded = torch.stack(expanded_patches, dim=1)  # (B, P, C_out, H)
        
        return expanded


class ChannelExpansionModule(nn.Module):
    """
    Expands 4-channel REVE embeddings to 8-channel representations.
    
    Uses position-based cross-attention followed by refinement layers.
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        position_dim: int = 3,
    ):
        """
        Args:
            hidden_dim: Dimension of REVE embeddings
            num_heads: Number of attention heads
            num_layers: Number of refinement layers
            dropout: Dropout rate
            position_dim: Dimension of electrode positions
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Main cross-attention for expansion
        self.cross_attention = PositionalCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            position_dim=position_dim,
        )
        
        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        input_positions: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand input embeddings to target channel count.
        
        Args:
            encoder_output: (batch, num_patches, num_input, hidden_dim)
                           e.g., (B, 11, 4, 512)
            input_positions: (batch, num_input, 3)
                            e.g., (B, 4, 3)
            target_positions: (batch, num_target, 3)
                             e.g., (B, 8, 3)
        
        Returns:
            expanded: (batch, num_patches, num_target, hidden_dim)
                     e.g., (B, 11, 8, 512)
        """
        B, P, C_in, H = encoder_output.shape
        C_out = target_positions.shape[1]
        
        # Cross-attention expansion
        expanded = self.cross_attention(
            encoder_output,
            input_positions,
            target_positions,
        )  # (B, P, C_out, H)
        
        # Refinement: process all patches and channels together
        # Reshape: (B, P, C_out, H) -> (B, P*C_out, H)
        refined = expanded.reshape(B, P * C_out, H)
        
        for layer in self.refinement_layers:
            refined = layer(refined)
        
        # Reshape back: (B, P*C_out, H) -> (B, P, C_out, H)
        expanded = refined.reshape(B, P, C_out, H)
        expanded = self.layer_norm(expanded)
        
        return expanded
