# -*- coding: utf-8 -*-
"""
Baseline 4: Spatiotemporal Transformer Encoder.
Uses global multi-head self-attention over flattened sensor time series.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import torch
import torch.nn as nn

class TransformerLocalizationModel(nn.Module):
    def __init__(self, in_features=11, hidden_dim=64, n_heads=4, num_layers=2, signal_length=100, graph_order=3):
        super().__init__()
        # Flatten the temporal footprint per sensor and project to hidden dimensions
        self.input_projection = nn.Linear(in_features * signal_length, hidden_dim)
        
        from graph_spectral_layer import PerFeatureGraphSpectralLayer
        self.graph_layer = PerFeatureGraphSpectralLayer(n_features=in_features, K=graph_order)
 
        self.input_projection = nn.Linear(in_features * signal_length, hidden_dim)
        
        # Standard Multi-Head Self-Attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.hidden_dim = hidden_dim
        
        # Continuous Projections Mapping to Intensity Field E(x)
        self.intensity_projector = nn.Linear(hidden_dim, signal_length)
        
        # Continuous Regression Tracking Heads
        self.coord_head = nn.Linear(hidden_dim, 2)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("t_grid", torch.linspace(0, 1, signal_length).unsqueeze(0))

    def forward(self, U, coords, lap_per_feature=None, adj_per_feature=None):
        """
        U: (B, N, T, F)
        coords: (B, N, 2)
        """
        B, N, T, F = U.shape
        current_device = U.device
        
        # 1. Flatten temporal steps and feature channels per station
        # Input shape transforms from (B, N, T, F) -> (B, N, T*F)
        x = U.reshape(B, N, T * F).contiguous()
        x = torch.relu(self.input_projection(x)) # Shape: (B, N, hidden_dim)
        
        # 2. Execute Global Multi-Head Self-Attention across all sensor tokens
        # Every station updates its vector by computing dependencies with all other stations
        transformer_out = self.transformer(x) # Shape: (B, N, hidden_dim)
        
        # 3. Reconstruct continuous E(x) intensity fields
        # Shape shifts from (B, N, hidden_dim) -> (B, N, T)
        intensity = self.intensity_projector(transformer_out).contiguous()
        
        # 4. Soft-argmax over time to predict continuous temporal onset E(t)
        temporal_profile = intensity.mean(dim=1) # (B, T)
        time_weights = torch.softmax(temporal_profile / 0.05, dim=1)
        time_pred = (time_weights * self.t_grid).sum(dim=1, keepdim=True) # (B, 1)
        
        # 5. Soft-argmax over sensors to predict continuous coordinates E(x)
        Ex = intensity.max(dim=2).values # (B, N)
        centroid     = coords.mean(dim=1, keepdim=True)                                    # (B, 1, 2)
        coords_c     = coords - centroid                                                    # (B, N, 2)
        scale        = coords_c.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)  # (B, 1, 1)
        coords_norm  = coords_c / scale                                                    # (B, N, 2)
        e_weights    = torch.softmax(Ex / 0.05, dim=1)
        pred_norm    = (e_weights.unsqueeze(-1) * coords_norm).sum(dim=1)                 # (B, 2)
        coords_pred_e = pred_norm * scale.squeeze(1) + centroid.squeeze(1)       
        
        # Generate dummy placeholders for explainability metrics to match Evaluator footprint
        dummy_score = torch.zeros((B, N), device=current_device)
        return (
            coords_pred_e, 
            time_pred, 
            intensity, 
            dummy_score, 
            dummy_score, 
            dummy_score, 
            dummy_score, 
            U.mean(dim=-1), 
            torch.zeros(1, device=current_device)
        )
