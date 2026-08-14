# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:48:38 2026
Updated: per-feature physics. Each feature (e.g. a different pollutant)
can now have its own sigma_space, velocity, broadening, and amplitude,
while all features share the same true source_location and
source_time -- modeling the realistic case where different
measurements of the same physical event propagate differently (e.g.
PM2.5 and NOx don't disperse identically), rather than just repeating
one signal across feature channels.

Backward compatible: passing scalars (as before) applies the same
physics to every feature, identical to the previous behavior for F=1.

@author: usman.anjum
"""

import numpy as np


class SyntheticEventGenerator:

    def __init__(
            self,
            n_sensors=20,
            n_timesteps=100,
            n_features=1,
            random_seed=42
    ):

        np.random.seed(random_seed)

        self.N = n_sensors
        self.T = n_timesteps
        self.F = n_features

    def generate_sensor_locations(self):

        coords = np.random.uniform(0.0, 1.0, (self.N, 2))

        return coords

    def _broadcast(self, param):
        """
        Accepts either a scalar (applied to every feature, the
        original behavior) or a sequence of length F (one value per
        feature). Returns a list of length F.
        """
        if np.isscalar(param):
            return [param] * self.F

        param = list(param)
        assert len(param) == self.F, (
            f"Expected {self.F} values (one per feature), got {len(param)}"
        )
        return param

    def generate_event(
            self,
            coords,
            source_location=None,
            source_time=None,
            sigma_space=0.15,
            sigma_time=8.0,
            amplitude=1.0,
            noise_std=0.05,
            velocity=0.05,
            broadening=2.0,
            decoy_location=None,
            decoy_boost=0.0,
            decoy_sigma=0.1,
    ):
        """
        All physics parameters may be a scalar (same for every
        feature) or a sequence of length F (per-feature physics). The
        true source location and time are always shared across
        features -- only HOW each feature's measurement propagates
        from that single shared source differs.

        decoy_location / decoy_boost / decoy_sigma (optional, default
        decoy_boost=0.0 means no decoy -- fully backward compatible):
        models a location, distinct from the true source, where
        amplitude gets artificially amplified (e.g. a valley where
        smoke physically accumulates, or a wind-convergence zone) --
        WITHOUT affecting arrival time or broadening, which remain
        governed purely by true distance from the source. This
        produces a sensor reading that is HIGH amplitude but still
        DELAYED and SMOOTHED (low high-frequency content), unlike a
        true source which is lower amplitude but SHARP and EARLY.
        This is the adversarial case naive intensity-based
        localization (E(x)) is expected to get fooled by, and that
        frequency-based localization (P(x)) is expected to resist.
        """

        if source_location is None:
            source_location = np.random.uniform(0, 1, 2)
        if source_time is None:
            source_time = np.random.randint(0, self.T)

        coords = np.array(coords)
        source_location = np.array(source_location)

        sigma_space_f = self._broadcast(sigma_space)
        sigma_time_f = self._broadcast(sigma_time)
        amplitude_f = self._broadcast(amplitude)
        velocity_f = self._broadcast(velocity)
        broadening_f = self._broadcast(broadening)

        distance = np.linalg.norm(coords - source_location, axis=1)  # (N,)
        t = np.arange(self.T)[None, :]  # (1, T)

        if decoy_location is not None:
            decoy_location = np.array(decoy_location)
            decoy_distance = np.linalg.norm(coords - decoy_location, axis=1)  # (N,)
            # Additive, independent of the source's own (often
            # near-zero at large distances) spatial decay -- a
            # multiplicative boost on top of an already-vanished
            # signal would still be ~zero. This models a genuinely
            # separate local accumulation effect.
            decoy_contribution = decoy_boost * np.exp(
                -decoy_distance ** 2 / (2 * decoy_sigma ** 2)
            )  # (N,)
        else:
            decoy_contribution = np.zeros(self.N)

        U = np.zeros((self.N, self.T, self.F))

        for f in range(self.F):
            spatial_component = np.exp(
                -distance ** 2 / (2 * sigma_space_f[f] ** 2)
            )  # (N,) -- governed by TRUE source distance only

            arrival_time = source_time + distance / velocity_f[f]  # (N,) -- TRUE source distance
            sigma_time_i = sigma_time_f[f] * (1.0 + broadening_f[f] * distance)  # TRUE source distance

            arrival_col = arrival_time[:, None]
            sigma_col = sigma_time_i[:, None]

            temporal_component = np.exp(
                -(t - arrival_col) ** 2 / (2 * sigma_col ** 2)
            )  # (N, T)

            total_spatial_amplitude = spatial_component + decoy_contribution  # (N,)

            U[:, :, f] = (
                amplitude_f[f] * total_spatial_amplitude[:, None] * temporal_component
            )

        U = U + np.random.normal(0, noise_std, U.shape)

        event_info = {
            "source_location": source_location,
            "source_time": source_time,
            "distance": distance,
            "decoy_location": decoy_location,
        }

        return U, event_info
