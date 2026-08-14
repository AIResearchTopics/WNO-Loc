# -*- coding: utf-8 -*-
"""
Baseline 3: Spatiotemporal Fourier Neural Operator (FNO1d).
Combines graph-spectral spatial layers with global Fourier frequency convolution heads.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import torch
import torch.nn as nn

class FNO1dTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Trainable complex weights to scale global Fourier modes
        # We store them as real components and view them as complex at runtime
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, 2) * 0.02
        )
        self.linear = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # Input shape x: (B*N, F, T) -> e.g., (251, 11, 100)
        B_N, F, T = x.shape
        
        # Fast Fourier Transform along the time dimension (dim=-1)
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Extract the active complex modes
        modes = min(self.modes, x_ft.size(-1))
        complex_weights = torch.view_as_complex(self.weights) # Shape: (F_in, F_out, modes)
        
        # Create a clean output tensor matching the target channel size (out_channels=32)
        # Shape must be: (Batch*Sensors, out_channels, total_fourier_frequencies)
        out_ft = torch.zeros(B_N, self.out_channels, x_ft.size(-1), device=x.device, dtype=x_ft.dtype)
        
        # FIXED EINSUM: Clear channel-mixing multiplication string matching dimensions perfectly
        # b: Batch*Sensors (251)
        # i: Input features/channels (11)
        # o: Output features/channels (32)
        # x: Active Fourier frequency modes (16)
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :modes], 
            complex_weights[:, :, :modes]
        )
        
        # Inverse Fast Fourier Transform back to the temporal domain
        x_fourier = torch.fft.irfft(out_ft, n=T, dim=-1)
        
        # Add the linear 1x1 convolution shortcut connection skip
        return torch.relu(x_fourier + self.linear(x))


class FNOLocalizationModel(nn.Module):
    def __init__(self, in_features=11, hidden_dim=32, modes=16, signal_length=100):
        super().__init__()
        # Preserves your exact Graph Spectral layer to maintain structural fairness!
        from graph_spectral_layer import PerFeatureGraphSpectralLayer
        self.graph_layer = PerFeatureGraphSpectralLayer(n_features=in_features, K=3)
        
        # 1D Fourier Operational Blocks
        self.fno1 = FNO1dTemporalBlock(in_features, hidden_dim, modes)
        self.fno2 = FNO1dTemporalBlock(hidden_dim, hidden_dim, modes)
        
        self.hidden_dim = hidden_dim
        
        # Continuous Projections Mapping to Intensity Field E(x)
        self.intensity_projector = nn.Linear(hidden_dim, 1)
        
        # Continuous Regression Tracking Heads
        self.coord_head = nn.Linear(hidden_dim, 2)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.register_buffer("t_grid", torch.linspace(0, 1, signal_length).unsqueeze(0))

    def forward(self, U, coords, lap_per_feature, adj_per_feature):
        """
        U: (B, N, T, F)
        coords: (B, N, 2)
        """
        B, N, T, F = U.shape
        current_device = U.device
        
        # Spatial mixing via graph layers
        U_mixed = self.graph_layer(U, lap_per_feature) # (B, N, T, F)
        
        # Shape transformations for 1D Fourier convolutions along time dimension
        x = U_mixed.reshape(B * N, T, F).permute(0, 2, 1).contiguous() # (B*N, F, T)
        
        # Run through global Fourier operators
        x = self.fno1(x)
        x = self.fno2(x) # Output shape: (B*N, hidden_dim, T)
        
        # Permute back to retrieve standard time-channel footprints
        x_out = x.permute(0, 2, 1).contiguous() # (B*N, T, hidden_dim)
        
        # Output the competing E(x) continuous intensity field 
        intensity = self.intensity_projector(x_out).squeeze(-1).reshape(B, N, T).contiguous()
        
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
            dummy_score, 
            dummy_score, 
            dummy_score, 
            dummy_score, 
            U.mean(dim=-1), 
            torch.zeros(1, device=current_device)
        )
