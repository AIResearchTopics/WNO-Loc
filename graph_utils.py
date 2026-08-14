# -*- coding: utf-8 -*-
"""
Graph construction utilities for the sensor network.

These build the spatial structure that has been missing from the
pipeline so far: a graph over sensors (nodes = sensors, edges = spatial
proximity), plus the rescaled graph Laplacian needed for spectral graph
filtering (the spatial analogue of the temporal wavelet decomposition
already implemented in wavelet_encoder.py).
"""

import torch


def build_knn_adjacency(coords, k=4):
    """
    coords: (N, 2) tensor or array-like
    Returns: binary symmetric adjacency (N, N) torch tensor.

    Builds a directed k-nearest-neighbor graph per node, then
    symmetrizes it via OR (edge i-j exists if i is among j's k nearest
    neighbors OR j is among i's), since a kNN graph is not naturally
    symmetric otherwise.
    """
    coords = torch.as_tensor(coords, dtype=torch.float32)
    N = coords.shape[0]

    dist = torch.cdist(coords, coords)        # (N, N)
    dist.fill_diagonal_(float('inf'))           # exclude self-matches

    k = min(k, N - 1)
    knn_idx = dist.topk(k, largest=False).indices  # (N, k)

    A = torch.zeros(N, N)
    for i in range(N):
        A[i, knn_idx[i]] = 1.0

    A = torch.maximum(A, A.T)  # symmetrize

    return A


def check_connectivity(A):
    """
    Returns True if the graph is fully connected (every node reachable
    from every other), via a simple BFS from node 0.

    This matters because spectral graph filtering propagates
    information through the graph structure -- if the graph has
    disconnected components, information can never flow between them
    no matter how the model is trained.
    """
    N = A.shape[0]
    visited = torch.zeros(N, dtype=torch.bool)
    stack = [0]
    visited[0] = True

    while stack:
        node = stack.pop()
        neighbors = torch.nonzero(A[node] > 0).flatten().tolist()
        for n in neighbors:
            if not visited[n]:
                visited[n] = True
                stack.append(n)

    return bool(visited.all().item())


def compute_rescaled_laplacian(A):
    """
    Symmetric normalized graph Laplacian, rescaled for Chebyshev
    polynomial spectral filtering:

        L_sym   = I - D^{-1/2} A D^{-1/2}
        L_tilde = (2 / lambda_max) * L_sym - I

    The symmetric normalized Laplacian's eigenvalues are guaranteed to
    lie in [0, 2] for any graph (a standard result in spectral graph
    theory -- Chung, 1997), so using lambda_max = 2 is an exact bound,
    not an approximation that could be wrong. This simplifies to:

        L_tilde = -D^{-1/2} A D^{-1/2}

    L_tilde has eigenvalues in [-1, 1], which is what the Chebyshev
    polynomial recursion (used in the upcoming graph wavelet layer)
    needs for numerical stability.
    """
    deg = A.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # (N, N)
    L_tilde = -A_norm

    return L_tilde
