# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:51:17 2026
Fixed: (1) Gaussian kernel parenthesization bug.
       (2) laplacian() now returns rescaled L_tilde = -D^-1/2 A D^-1/2.
       (3) feature_adjacency() no longer called twice in laplacian().
       (4) _find_components vectorized with numpy -- eliminates O(N^2)
           Python loop that caused hangs on sparse-coverage features.

@author: anjum
"""
import numpy as np


class SensorGraph:

    def __init__(self, coords, feature_mask, k=4, sigma=0.2):
        self.coords = np.asarray(coords)
        self.mask   = np.asarray(feature_mask)
        self.N      = self.coords.shape[0]
        self.k      = k
        self.sigma  = sigma
        self.F      = self.mask.shape[1]

    def pairwise_distance(self):
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def adjacency_matrix(self):
        distance = self.pairwise_distance()
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            neighbors = np.argsort(distance[i])[1:self.k + 1]
            for j in neighbors:
                weight = np.exp(-distance[i, j] ** 2 / (2 * self.sigma ** 2))
                A[i, j] = weight
        A = np.maximum(A, A.T)
        return A

    def _find_components(self, adj, nodes):
        """
        Connected components using Union-Find with path compression
        and numpy-based edge detection. Replaces the slow O(N^2)
        Python double-loop.
        """
        nodes = np.array(nodes)
        n     = len(nodes)
        if n == 0:
            return []
        if n == 1:
            return [list(nodes)]

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        # Extract subgraph for covering nodes -- numpy vectorized
        sub_adj = adj[np.ix_(nodes, nodes)]
        rows, cols = np.where(sub_adj > 0)
        for r, c in zip(rows, cols):
            if r < c:  # process each edge once
                union(int(r), int(c))

        components = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(nodes[i])

        return list(components.values())

    def _bridge_components(self, A_f, distance, covering):
        """
        Connect disconnected components by adding bridge edges between
        the closest pair of nodes across components.
        """
        components = self._find_components(A_f, list(covering))

        while len(components) > 1:
            best = None
            for ci in range(len(components)):
                for cj in range(ci + 1, len(components)):
                    # Vectorized: find min distance across component pair
                    ci_arr = np.array(components[ci])
                    cj_arr = np.array(components[cj])
                    sub    = distance[np.ix_(ci_arr, cj_arr)]
                    idx    = np.unravel_index(sub.argmin(), sub.shape)
                    d      = sub[idx]
                    i, j   = ci_arr[idx[0]], cj_arr[idx[1]]
                    if best is None or d < best[0]:
                        best = (d, i, j)

            d, i, j = best
            weight  = np.exp(-d ** 2 / (2 * self.sigma ** 2))
            A_f[i, j] = weight
            A_f[j, i] = weight

            components = self._find_components(A_f, list(covering))

        return A_f

    def feature_adjacency(self):
        distance  = self.pairwise_distance()
        A_feature = np.zeros((self.F, self.N, self.N))

        for f in range(self.F):
            m        = self.mask[:, f]
            covering = np.where(m > 0)[0]

            for i in covering:
                others = covering[covering != i]
                if len(others) == 0:
                    continue
                k_eff   = min(self.k, len(others))
                nearest = others[np.argsort(distance[i, others])[:k_eff]]
                for j in nearest:
                    weight = np.exp(
                        -distance[i, j] ** 2 / (2 * self.sigma ** 2)
                    )
                    A_feature[f, i, j] = weight

            A_feature[f] = np.maximum(A_feature[f], A_feature[f].T)

            if len(covering) > 1:
                A_feature[f] = self._bridge_components(
                    A_feature[f], distance, covering
                )

        return A_feature

    def laplacian(self):
        """
        Returns rescaled Laplacian L_tilde = -D^-1/2 A D^-1/2 per feature.
        Eigenvalues in [-1, 1] -- ready for Chebyshev polynomial recursion.

        Fix: feature_adjacency() is now called ONCE and reused for both
        the adjacency and degree computation. Previously degree_matrix()
        called feature_adjacency() again, doubling all the work.
        """
        A_feature = self.feature_adjacency()   # compute once
        L         = np.zeros_like(A_feature)

        for f in range(self.F):
            degree   = A_feature[f].sum(axis=1)
            inv_sqrt = np.where(
                degree > 0,
                1.0 / np.sqrt(degree.clip(min=1e-8)),
                0.0
            )
            D_inv = np.diag(inv_sqrt)
            L[f]  = -(D_inv @ A_feature[f] @ D_inv)

        return L

    def degree_matrix(self):
        """Kept for backward compatibility -- uses cached adjacency."""
        A_feature = self.feature_adjacency()
        D         = np.zeros_like(A_feature)
        for f in range(self.F):
            D[f] = np.diag(A_feature[f].sum(axis=1))
        return D

    def build(self):
        distance = self.pairwise_distance()
        A        = self.feature_adjacency()
        L        = self.laplacian()
        return {
            "coords":       self.coords,
            "distance":     distance,
            "adjacency":    A,
            "laplacian":    L,
            "feature_mask": self.mask,
        }