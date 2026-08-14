# -*- coding: utf-8 -*-
"""
Baseline 2: Spatiotemporal Bidirectional LSTM.
Combines graph-spectral spatial layers with recurrent sequence encoders.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import torch
import torch.nn as nn

class LSTMIntensityBaseline(nn.Module):
    def __init__(self, in_features=11, hidden_dim=32, num_layers=2, signal_length=100):
        super().__init__()
        # 1. Spatial Channel Regularizer: Preserves your exact Graph Spectral Layer
        from graph_spectral_layer import PerFeatureGraphSpectralLayer
        self.graph_layer = PerFeatureGraphSpectralLayer(n_features=in_features, K=3)
        
        # 2. Temporal Sequence Block: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_out_dim = hidden_dim * 2
        self.hidden_dim = hidden_dim
        self.lstm_out_dim = lstm_out_dim
        
        # 3. Continuous Projections Mapping to Intensity Field E(x)
        self.intensity_projector = nn.Linear(lstm_out_dim, 1)
        
        # 4. Continuous Regression Tracking Heads
        self.coord_head = nn.Linear(lstm_out_dim, 2)
        self.time_head = nn.Linear(lstm_out_dim, 1)
        self.register_buffer("t_grid", torch.linspace(0, 1, signal_length).unsqueeze(0))

    def forward(self, U, coords, lap_per_feature, adj_per_feature):
        """
        U: (B, N, T, F)
        coords: (B, N, 2)
        """
        B, N, T, F = U.shape
        current_device = U.device
        
        # Pass data through your custom spatial graph layers first
        U_mixed = self.graph_layer(U, lap_per_feature) # Output: (B, N, T, F)
        
        # Flatten batches and nodes to feed them as independent recurrent timelines
        x = U_mixed.reshape(B * N, T, F).contiguous() # (B*N, T, F)
        
        # Run through the LSTM sequence engine
        # lstm_out shape: (B*N, T, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Output the competing E(x) continuous intensity field 
        intensity = self.intensity_projector(lstm_out).squeeze(-1).reshape(B, N, T).contiguous()
        
        # Soft-argmax over time to predict continuous temporal onset E(t)
        temporal_profile = intensity.mean(dim=1) # (B, T)
        time_weights = torch.softmax(temporal_profile / 0.05, dim=1)
        time_pred = (time_weights * self.t_grid).sum(dim=1, keepdim=True) # (B, 1)
        
        # Soft-argmax over sensors to predict continuous coordinates E(x)
        Ex = intensity.max(dim=2).values # (B, N)
        centroid     = coords.mean(dim=1, keepdim=True)                                    # (B, 1, 2)
        coords_c     = coords - centroid                                                    # (B, N, 2)
        scale        = coords_c.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)  # (B, 1, 1)
        coords_norm  = coords_c / scale                                                    # (B, N, 2)
        e_weights    = torch.softmax(Ex / 0.05, dim=1)
        pred_norm    = (e_weights.unsqueeze(-1) * coords_norm).sum(dim=1)                 # (B, 2)
        coords_pred_e = pred_norm * scale.squeeze(1) + centroid.squeeze(1)       
        
        # Generate dummy placeholders for explainability metrics to match Evaluator's return footprint
        dummy_score = torch.zeros((B, N), device=current_device)
        return (
            coords_pred_e, 
            time_pred, 
            intensity, 
            dummy_score, # Px placeholder
            dummy_score, # Frequency score placeholder
            dummy_score, # Temporal score placeholder
            dummy_score, # Neighbor score placeholder
            U.mean(dim=-1), # Band energy placeholder
            torch.zeros(1, device=current_device) # Alpha placeholder
        )


