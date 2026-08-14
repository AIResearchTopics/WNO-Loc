# -*- coding: utf-8 -*-
"""
Wavelet Neural Operator (WNO) -- 1D version adapted from the official
Tripura & Chakraborty (2023) implementation at:
  https://github.com/TapasTripura/WNO

Key differences from what was here before
------------------------------------------
The previous wavelet_encoder.py used a hand-coded Haar strided
convolution (F.conv1d with a hard-coded 2-tap filter) and F.interpolate
for reconstruction -- this is NOT a real WNO. Specifically:

1. The real WNO uses `DWT1D` / `IDWT1D` from the `pytorch_wavelets`
   library, which implements a proper perfect-reconstruction discrete
   wavelet transform for any wavelet (db1, db4, sym4, ...) -- not a
   Haar approximation.

2. The real WNO applies learned weights to BOTH the approximation
   coefficients (weights1) AND the highest-level detail coefficients
   (weights2) -- the previous version incorrectly spread learned
   weights across all sub-bands using linear layers, which is closer
   to a wavelet scattering transform than a WNO.

3. The reconstruction uses proper IDWT1D, not F.interpolate, which is
   NOT an inverse wavelet transform (it doesn't satisfy the
   perfect-reconstruction property).

4. The real WNO concatenates the spatial grid (x-coordinates) with the
   input, giving the operator explicit positional information -- this
   is critical for learning spatially-varying dynamics and was missing.

This file implements exactly what the official code does, adapted to
work with sensor network data (B, N, T, F) instead of PDE grids (B, x).
"""

"""
Optimized Wavelet Neural Operator (WNO) -- 1D version.
Fixes applied:
  - Ensured explicit device-agnostic safety for cached DWT blocks.
  - Removed dangerous pre-allocated zero tensors causing potential shape crashes.
  - Refactored grid extraction as a persistent structural buffer inside WNO1d.
  - Applied memory contiguity strides to maximize compatibility across CPU, CUDA, and XPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D


class WaveConv1d(nn.Module):
    """
    1D Wavelet kernel integral layer.
    """

    def __init__(self, in_channels, out_channels, level, size,
                 wavelet='db4', mode='symmetric'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size
        self.wavelet = wavelet
        self.mode = mode

        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode=self.mode)

        dummy = torch.randn(1, 1, self.size)
        mode_data, _ = self.dwt_(dummy)
        self.modes1 = mode_data.shape[-1]

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1))

        self.idwt_ = IDWT1D(wave=self.wavelet, mode=self.mode)
        self._dwt_cache = {self.level: self.dwt_}

    def mul1d(self, inp, weights):
        # Explicitly enforce contiguous layouts for optimized XPU/NVIDIA matrix sweeps
        return torch.einsum("bix,iox->box", inp.contiguous(), weights)

    def forward(self, x):
        target_device = x.device  # Track active target device dynamically
        
        # Handle resolution mismatch seamlessly
        if x.shape[-1] == self.size:
            dwt = self.dwt_
        else:
            if x.shape[-1] > self.size:
                factor = int(np.log2(x.shape[-1] // self.size))
                level = self.level + factor
            else:
                factor = int(np.log2(self.size // x.shape[-1]))
                level = self.level - factor
                
            if level not in self._dwt_cache:
                self._dwt_cache[level] = DWT1D(wave=self.wavelet, J=level, mode=self.mode)
            dwt = self._dwt_cache[level]

        # DEVICE AGNOSTIC PROTECTION: Ensure transform modules point to the current active device VRAM
        dwt.to(target_device)
        self.idwt_.to(target_device)

        x_ft, x_coeff = dwt(x)

        # Rebuild out_coeff list to completely avoid variable shape mismatches during projection
        out_coeff = [torch.zeros_like(c) for c in x_coeff]

        # Apply learned weights directly to approximation and finest detail frequencies
        out_ft = self.mul1d(x_ft, self.weights1)
        out_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)

        return self.idwt_((out_ft, out_coeff))


class WNO1d(nn.Module):
    """
    Full 1D Wavelet Neural Operator.
    """

    def __init__(self, width, level, layers, size, wavelet,
                 in_channels, out_channels, grid_range=1.0, padding=0):
        super().__init__()

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.grid_range = grid_range
        self.padding = padding

        # Lift inputs (+1 for spatial grid coordinate concatenated below)
        self.fc0 = nn.Linear(in_channels + 1, self.width)

        self.conv = nn.ModuleList([
            WaveConv1d(self.width, self.width, self.level, self.size, self.wavelet)
            for _ in range(self.layers)
        ])
        self.w = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
            for _ in range(self.layers)
        ])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # DEVICE-AGNOSTIC BUFFER: Pre-instantiated grid data shifts automatically with model.to()
        self.register_buffer(
            "base_grid", 
            torch.linspace(0, self.grid_range, self.size).reshape(1, self.size, 1)
        )

    def get_grid(self, batch_size, time_steps):
        # Expand our cached buffer safely across different inference batches
        # Handles dynamic testing resolutions if time_steps vary from self.size
        if time_steps == self.size:
            return self.base_grid.expand(batch_size, -1, -1)
        else:
            # Fallback path if resolution changes during testing phase
            alt_grid = torch.linspace(0, self.grid_range, time_steps, device=self.base_grid.device)
            return alt_grid.reshape(1, time_steps, 1).expand(batch_size, -1, -1)

    def forward(self, x):
        # x shape: (B, T, in_channels)
        B, T, _ = x.shape
        grid = self.get_grid(B, T)
        
        x = torch.cat([x, grid], dim=-1)     # (B, T, in_channels+1)
        x = self.fc0(x)                       # (B, T, width)
        x = x.permute(0, 2, 1).contiguous()   # (B, width, T) for contiguous convolution sweeps

        if self.padding:
            x = F.pad(x, [0, self.padding])

        for i, (conv, w) in enumerate(zip(self.conv, self.w)):
            x = conv(x) + w(x)
            if i < self.layers - 1:
                x = F.mish(x)

        if self.padding:
            x = x[..., :-self.padding]

        x = x.permute(0, 2, 1).contiguous()   # (B, T, width)
        x = F.gelu(self.fc1(x))              # (B, T, 128)
        x = self.fc2(x)                       # (B, T, out_channels)
        return x

