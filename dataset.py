# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:06:36 2026
Updated: supports multi-feature, partial sensor coverage. If
feature_mask is not provided and F>1, a random partial-coverage mask
is generated automatically (every sensor measures >=1 feature, every
feature is measured by >=1 sensor, to avoid degenerate empty graphs).
Raw generated values are zeroed out wherever a sensor doesn't measure
that feature, before being passed to the model.

@author: anjum
"""

import numpy as np
import torch

from torch.utils.data import Dataset
from sensor_graph import SensorGraph
from graph_utils import check_connectivity


def generate_feature_mask(N, F, coverage_prob=0.6, seed=None):
    """
    Random binary (N, F) coverage mask: mask[i, f] = 1 if sensor i
    measures feature f. Ensures no sensor is fully isolated (measures
    nothing) and no feature has zero coverage (which would make its
    graph empty and meaningless).
    """
    rng = np.random.RandomState(seed)
    mask = (rng.rand(N, F) < coverage_prob).astype(float)

    for i in range(N):
        if mask[i].sum() == 0:
            mask[i, rng.randint(F)] = 1

    for f in range(F):
        if mask[:, f].sum() == 0:
            mask[rng.randint(N), f] = 1

    return mask


class EventDataset(Dataset):
    """
    Each sample is a 5-tuple:
        (U, coords, laplacian_per_feature, location, time)

    U has unmeasured (sensor, feature) entries zeroed out according to
    feature_mask, BEFORE being handed to the model -- the model only
    ever sees real measurements or structural zeros, never values a
    sensor never actually produced.
    """

    def __init__(
            self,
            generator,
            num_events=100,
            coords=None,
            lazy=False,
            k_neighbors=4,
            graph_sigma=0.2,
            feature_mask=None,
            coverage_prob=0.6,
            mask_seed=0,
            real_U=None,
            real_locations=None,
            precomputed_lap=None, 
            precomputed_adj=None,
            verbose=False,
            **event_kwargs
    ):

        self.generator = generator
        self.num_events = num_events
        self.event_kwargs = event_kwargs
        self.lazy = lazy

        self.coords = coords if coords is not None else generator.generate_sensor_locations()
        self.coords_tensor = torch.tensor(self.coords, dtype=torch.float32)
        
        N = self.coords.shape[0]

        if real_U is not None:
            F = real_U.shape[3]
        else:
            F = generator.F

        if feature_mask is None:
            if F == 1:
                feature_mask = np.ones((N, F))
            else:
                feature_mask = generate_feature_mask(
                    N, F, coverage_prob, seed=mask_seed
                )
        self.feature_mask = feature_mask

        if precomputed_lap is not None and precomputed_adj is not None:
            # Use pre-computed graph from cache -- skip expensive k-NN
            self.laplacian_per_feature = torch.tensor(
                precomputed_lap, dtype=torch.float32
            )
            self.adjacency_per_feature = torch.tensor(
                precomputed_adj, dtype=torch.float32
            )
        else:
            # Build graph from scratch (synthetic data or missing cache)
            self.sensor_graph = SensorGraph(
                self.coords, feature_mask,
                k=k_neighbors, sigma=graph_sigma
            )
            L = self.sensor_graph.laplacian()
            A = self.sensor_graph.feature_adjacency()
            self.laplacian_per_feature = torch.tensor(L, dtype=torch.float32)
            self.adjacency_per_feature = torch.tensor(A, dtype=torch.float32)

            if verbose:
                for f in range(F):
                    covering = np.where(feature_mask[:, f] > 0)[0]
                    coverage = len(covering)
                    if coverage <= 1:
                        print(f"  feature {f}: coverage={coverage}/{N} "
                              f"sensors (too few)")
                        continue
                    sub_adj   = self.adjacency_per_feature[f][covering][:, covering]
                    connected = check_connectivity(sub_adj)
                    status    = "connected" if connected else "NOT connected"
                    print(f"  feature {f}: coverage={coverage}/{N} sensors, "
                          f"graph among covering sensors is {status}")

        # for f in range(F):
        #     covering = np.where(feature_mask[:, f] > 0)[0]
        #     coverage = len(covering)
        #     if coverage <= 1:
        #         print(f"  feature {f}: coverage={coverage}/{N} sensors (too few to check connectivity)")
        #         continue

        #     sub_adjacency = self.adjacency_per_feature[f][covering][:, covering]
        #     connected = check_connectivity(sub_adjacency)
        #     status = "connected" if connected else "NOT connected"
        #     print(f"  feature {f}: coverage={coverage}/{N} sensors, graph among covering sensors is {status}")

        if real_U is not None and real_locations is not None:
            self.lazy = False  # Ensure lazy evaluation is turned off for pre-loaded files
            self.samples = []
            
            for i in range(len(real_U)):
                # Apply your feature mask to mask out non-reporting stations
                masked_U = real_U[i] * self.feature_mask[:, None, :]
                
                # Bundle into the exact 5-tuple payload box your trainer expects!
                self.samples.append((
                    torch.tensor(masked_U, dtype=torch.float32),
                    self.coords_tensor,
                    self.laplacian_per_feature,
                    self.adjacency_per_feature,
                    torch.tensor(real_locations[i], dtype=torch.float32),
                    torch.tensor([48.0 / real_U.shape[2]], dtype=torch.float32) # Event time onset
                ))
        elif not self.lazy:
            self.samples = [self._make_sample() for _ in range(num_events)]        

    def _make_sample(self):

        U, event_info = self.generator.generate_event(self.coords, **self.event_kwargs)

        # zero out entries where the sensor doesn't measure that feature
        U = U * self.feature_mask[:, None, :]

        location = torch.tensor(event_info["source_location"], dtype=torch.float32)

        time = torch.tensor(
            [event_info["source_time"] / self.generator.T], dtype=torch.float32
        )

        U_tensor = torch.tensor(U, dtype=torch.float32)

        return (
            U_tensor,
            self.coords_tensor,
            self.laplacian_per_feature,
            self.adjacency_per_feature,
            location,
            time,
        )

    def __len__(self):
        """
        Dynamically returns the true size of the dataset pool.
        Prevents IndexError bugs by verifying real memory sizes.
        """
        if hasattr(self, 'samples') and self.samples is not None:
            # If the pre-computed sample list exists (Real data path or non-lazy synthetic),
            # return its exact actual populated item count!
            return len(self.samples)
            
        # Fall back to your original synthetic parameter defaults if running in pure lazy mode
        return self.num_events

    def __getitem__(self, idx):

        if self.lazy:
            return self._make_sample()

        return self.samples[idx]
