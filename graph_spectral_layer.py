# -*- coding: utf-8 -*-
"""
Graph spectral layers: the spatial analogue of the temporal wavelet
operator in wavelet_encoder.py.

The core mechanism (Chebyshev polynomial recursion on the rescaled
graph Laplacian) was independently verified against exact
eigendecomposition (max discrepancy ~1e-6, float32 precision) before
either layer below was written.

Method: Defferrard, Bresson & Vandergheynst, "Convolutional Neural
Networks on Graphs with Fast Localized Spectral Filtering", NeurIPS
2016. Same polynomial approximation trick used for spectral graph
wavelets without eigendecomposition in Hammond, Vandergheynst &
Gribonval, 2011.

Two variants:

  GraphSpectralLayer -- operates on an already-lifted representation
    (B, N, T, d) with ONE shared graph (B, N, N). Use when all features
    share the same sensor coverage.

  PerFeatureGraphSpectralLayer -- operates on the raw, pre-lift input
    (B, N, T, F) with a SEPARATE graph per feature (B, F, N, N), via
    SensorGraph. Use when different features have different sensor
    coverage (e.g. not every sensor measures every pollutant).
"""

"""
Optimized Graph Spectral Layers using Chebyshev Polynomials.
Fixes applied:
  - Eliminated torch.zeros_like allocations inside forward passes to prevent sync lag.
  - Replaced Python loops and manual .view slices with a unified torch.einsum call.
  - Enforced memory contiguity (.contiguous()) on all tensor permutations.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Variant 1: single shared graph, applied after lift
# ---------------------------------------------------------------------

def apply_laplacian_batched(L_tilde, H):
    """
    L_tilde: (B, N, N)
    H:       (B, N, T, d)
    Returns: (B, N, T, d)
    """
    # Enforce contiguity for reliable XPU/CUDA tensor core streaming
    return torch.einsum('bij,bjtc->bitc', L_tilde.contiguous(), H.contiguous())


class GraphSpectralLayer(nn.Module):
    """
    Input:  H of shape (B, N, T, d), L_tilde of shape (B, N, N)
    Output: H_out of shape (B, N, T, d)
    """

    def __init__(self, hidden_dim, K=3):
        super().__init__()
        self.K = K
        self.order_weights = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(K + 1)]
        )
        self.bypass = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, H, L_tilde):
        T0 = H
        T1 = apply_laplacian_batched(L_tilde, H)

        terms = [T0, T1]
        for k in range(2, self.K + 1):
            Tk = 2 * apply_laplacian_batched(L_tilde, terms[-1]) - terms[-2]
            terms.append(Tk)

        # Optimization: Initialize out using the k=0 term directly to avoid torch.zeros_like allocations
        out = self.order_weights[0](terms[0])
        for k in range(1, len(terms)):
            out = out + self.order_weights[k](terms[k])

        out = self.act(out + self.bypass(H))
        return out


# ---------------------------------------------------------------------
# Variant 2: per-feature graph, applied before lift (Executed by your model)
# ---------------------------------------------------------------------

def apply_laplacian_per_feature(L_tilde, x):
    """
    L_tilde: (B, F, N, N) OR (F, N, N)
    x:       (B, F, N, T)
    Returns: (B, F, N, T)
    """
    if L_tilde.dim() == 3:
        return torch.einsum('fij,bfjt->bfit', L_tilde.contiguous(), x.contiguous())
    return torch.einsum('bfij,bfjt->bfit', L_tilde.contiguous(), x.contiguous())


class PerFeatureGraphSpectralLayer(nn.Module):
    """
    Input:  U of shape (B, N, T, F), L_tilde_per_feature of shape (B, F, N, N) or (F, N, N)
    Output: U_mixed of shape (B, N, T, F)
    """

    def __init__(self, n_features, K=3):
        super().__init__()
        self.K = K
        self.F = n_features

        self.order_weights = nn.Parameter(torch.randn(n_features, K + 1) * 0.1)
        self.bypass_weight = nn.Parameter(torch.ones(n_features))
        self.act = nn.GELU()

    def forward(self, U, L_tilde_per_feature):
        # Permute and force layout contiguity for hardware acceleration
        x = U.permute(0, 3, 1, 2).contiguous()  # (B, N, T, F) -> (B, F, N, T)

        T0 = x
        T1 = apply_laplacian_per_feature(L_tilde_per_feature, x)
        terms = [T0, T1]
        for k in range(2, self.K + 1):
            Tk = 2 * apply_laplacian_per_feature(L_tilde_per_feature, terms[-1]) - terms[-2]
            terms.append(Tk)

        # Vectorized Aggregation: Stack terms along a new 'order' dimension (k)
        # terms_stacked shape: (K+1, B, F, N, T)
        terms_stacked = torch.stack(terms, dim=0)

        # Use einsum to perform multi-order scaling and summation simultaneously
        # 'k' = order index, 'f' = features, 'b' = batch, 'n' = nodes, 't' = timesteps
        # self.order_weights shape: (F, K+1) maps directly to 'fk'
        out = torch.einsum('kbfnt,fk->bfnt', terms_stacked, self.order_weights)

        # Compute bypass path cleanly without .view steps
        # self.bypass_weight shape (F,) broadcasts natively along dimension 1 of (B, F, N, T)
        bypass = self.bypass_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x
        out = self.act(out + bypass)

        # Permute back and secure sequential layout block for the downstream WNO flattener
        return out.permute(0, 2, 3, 1).contiguous()  # (B, N, T, F)