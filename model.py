# -*- coding: utf-8 -*-
"""
Full localization model using the real WNO.

Pipeline:
  1. PerFeatureGraphSpectralLayer -- spatial mixing across sensors,
     per feature, using the per-feature graph from SensorGraph.
     (B, N, T, F) -> (B, N, T, F)

  2. Per-sensor WNO1d -- temporal operator for each sensor independently.
     The WNO (Tripura & Chakraborty 2023) maps the F-feature time series
     at each sensor to a scalar event intensity time series.
     Input per sensor: (B, T, F) -> Output: (B, T, 1)

  3. Event intensity field: (B, N, T) -- E(x) from the methodology.

  4. Source activation score P(x): high-frequency ratio from the WNO's
     own wavelet coefficients -- used alongside E(x) for comparison.

  5. Localization via soft-argmax over E(x) and P(x).

@author: anjum
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from wavelet_encoder import WNO1d
from graph_spectral_layer import PerFeatureGraphSpectralLayer
from pytorch_wavelets import DWT1D


class LocalizationModel(nn.Module):

    def __init__(
            self,
            in_features=1,
            use_real_data=False,
            wno_width=32,
            wno_levels=3,
            wno_layers=4,
            signal_length=100,
            wavelet='db4',
            graph_order=3,
            temperature=0.05,
    ):
        super().__init__()

        self.temperature = temperature
        self.wno_levels = wno_levels
        self.wavelet = wavelet
        self.in_features = in_features
        self.n_bands = self.wno_levels + 1 
        self.use_real_data = use_real_data
        
        self.band_weights = nn.Parameter(torch.ones(self.wno_levels + 1))

        # Pre-instantiate the wavelet operator once to save creation overhead
        self.band_energy_dwt = DWT1D(wave=self.wavelet, J=self.wno_levels, mode='symmetric')

        # DEVICE-AGNOSTIC BUFFER: This automatically moves to CPU/CUDA/XPU when model.to() is called
        self.register_buffer("t_grid", torch.linspace(0, 1, signal_length).unsqueeze(0))  # Shape: (1, T)

        # Per-feature spatial mixing via graph spectral layer
        self.graph_layer = PerFeatureGraphSpectralLayer(n_features=in_features, K=graph_order)

        # Per-sensor WNO1d (temporal operator)
        self.wno = WNO1d(
            width=wno_width, level=wno_levels, layers=wno_layers,
            size=signal_length, wavelet=wavelet, in_channels=in_features, out_channels=1,
        )
        
        self.log_temp_t = nn.Parameter(torch.tensor([-3.2]))  # dynamic time temperature
        
        self.log_temp_spatial = nn.Parameter(torch.tensor([-3.2])) # Dynamic spatial temperature

        self.px_head = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Softplus() # Forces attribution scores to be strictly positive
        )
        
        # new addition
        self.graph_refinement = PerFeatureGraphSpectralLayer(
            n_features=1,
            K=graph_order
        )
        
        self.localization_head = nn.Sequential(
            nn.Linear(3,16),
            nn.GELU(),
            nn.Linear(16,1)
        )
        
        self.log_tau = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def _compute_band_energy(self, U_flat, device):
        """
      Compute per-sensor wavelet band energies and derived signals.

      U_flat: (B*N, T, F)

      Returns:
          Px_flat:    (B*N,) -- learnable-weighted sum of band energies
                      = a0*E_approx + a1*E_D1 + a2*E_D2 + ... + aJ*E_DJ
                      Higher = more energy in bands the model finds
                      informative for source proximity.

          entropy_flat: (B*N,) -- Shannon entropy of band energy distribution.
                        Low = energy concentrated in few bands = sharp
                        transient = likely close to source.
                        High = spread across bands = smooth propagated
                        signal = downstream of source.

          band_energy: (B*N, n_bands) -- raw per-band energy values,
                         used for logging the learned a_k weights and
                         for the explainability attribution output.
      """
        # Permute to (B*N, F, T) and force memory contiguity for hardware acceleration
        x = U_flat.permute(0, 2, 1).contiguous()       

        # Ensure the DWT tracking matches the active execution device, 
        # DWT: approx (B*N, F, T_A) and list of details (B*N, F, T_D) per level
        self.band_energy_dwt.to(device)
        approx, details = self.band_energy_dwt(x)

        # Low frequency energy, summed across features and time (B*N)
        low_energy_components = approx.pow(2).sum(dim=(1, 2)) #(B*N,)
        # Individual energy components (high energy) summed across features and time (B*N)
        high_energy_components = [d.pow(2).sum(dim=(1, 2)) for d in details] # (B*N, wno_levels + 1)
        
        # Stack all bands: [approx, detail_1, ..., detail_J] (B*N, wno_layers-1)
        band_energy = torch.stack([low_energy_components] + high_energy_components, dim=1)
        
        # Learnable weighted sum using softmax - weights sum to 1
        # band_weights shape: (n_bands,) -> broadcast over (B*N, n_bands)
        # alpha = torch.softmax(self.band_weights, dim=0) # (n_bands,)
        alpha_detail = torch.softmax(self.band_weights[1:], dim=0) # (n_bands,)
        zero_approx = torch.zeros(1, device=band_energy.device, dtype=band_energy.dtype)
        alpha = torch.cat([zero_approx, alpha_detail])
        # alpha_approx = torch.sigmoid(self.band_weights[0:1]) * 0.01  # small leaky approximation
        # alpha = torch.cat([alpha_approx, alpha_detail])

        band_energy_norm = band_energy / (band_energy.sum(dim=1, keepdim=True) + 1e-8)
        Px_flat = (band_energy_norm * alpha.unsqueeze(0)).sum(dim=1) # (B*N,)
        
        # Shanon entroy of normalized band energy distribution. This tells us which band dominates
        total_energy = band_energy.sum(dim=1, keepdim=True) + 1e-8
        probs = band_energy / total_energy # (B*N, n_bands)
        probs_safe = probs.clamp(min=1e-8)
        Hx_flat = -(probs_safe * probs_safe.log()).sum(dim=1) # (B*N,)
        
        return Px_flat, Hx_flat, band_energy
    
    def forward(self, U, coords, lap_per_feature, adj_per_feature):
        """
        U:               (B, N, T, F)
        coords:          (B, N, 2)
        lap_per_feature: (F, N, N) or (B, F, N, N)
        adj_per_feature: (F, N, N) or (B, F, N, N)
 
        Returns:
            coords_pred_e: (B, 2)    -- E(x)-based localization
            time_pred:     (B, 1)    -- predicted event onset time
            intensity:     (B, N, T) -- event intensity field E(x)
            Px:            (B, N)    -- learnable-weighted band energy P(x)
            entropy:       (B, N)    -- H(x) wavelet entropy per sensor
        """
        B, N, T, Feat = U.shape
        current_device = U.device  # Read the runtime device dynamically

        # Spatial mixing per feature
        U_mixed = self.graph_layer(U, lap_per_feature)  # (B, N, T, F)

        # Per-sensor WNO1d temporal operator
        U_flat = U_mixed.reshape(B * N, T, Feat).contiguous()           
        intensity_flat = self.wno(U_flat)            # (B*N, T, 1)                
        intensity = intensity_flat.squeeze(-1).reshape(B, N, T).contiguous()  # (B, N, T)
        
        if adj_per_feature.dim() == 3:
        # (F,N,N)
            adj_matrix = adj_per_feature.mean(dim=0)
        else:
        # (B,F,N,N)
            adj_matrix = adj_per_feature.mean(dim=1)
        
        # P(x), entropy, raw band energies from graph-mixed signal
        Px_flat, Hx_flat, band_energy_flat = self._compute_band_energy(U_flat, current_device)                        
        # Hx = Hx_flat.reshape(B, N) # (B, N)
        band_energy = band_energy_flat.reshape(B, N, self.n_bands) # (B, N, n_bands)
        degree = adj_matrix.sum(dim=-1, keepdim=True) + 1e-8
        neighbor_energy = torch.bmm(adj_matrix, band_energy)

        neighbor_energy = neighbor_energy / degree
        # contrast = band_energy - neighbor_energy
        band_energy_norm = band_energy / (band_energy.sum(dim=-1, keepdim=True) + 1e-8)
        neighbor_avg_norm = torch.bmm(adj_matrix, band_energy_norm) / degree
        contrast = F.relu(band_energy_norm - neighbor_avg_norm)
        contrast = torch.relu(contrast)
        # contrast = contrast / (contrast.mean(dim=1, keepdim=True) + 1e-8)
        tau = torch.clamp(torch.exp(self.log_tau), min=0.1, max=2.0)
        
        # alpha_detail = torch.softmax(self.band_weights[1:], dim=0)
        
        alpha_detail = torch.softmax(self.band_weights[1:] / tau, dim=0)        

        alpha = torch.cat([torch.zeros(1, device=band_energy.device, dtype=alpha_detail.dtype), alpha_detail])
        
        # alpha = torch.softmax(self.band_weights / tau, dim=0)

        # E(x), Soft-argmax over time 
        temporal_profile = intensity.max(dim=1).values #(B, T)
        t_temp = torch.exp(self.log_temp_t) + 1e-4
        time_weights = torch.softmax(temporal_profile / t_temp, dim=1)
        time_pred = (time_weights * self.t_grid).sum(dim=1, keepdim=True)

        # time_weights = torch.softmax(temporal_profile / self.temperature, dim=1)
        # time_pred = (time_weights * self.t_grid).sum(dim=1, keepdim=True)  # (B, 1)

        # Soft-argmax over sensors using raw E(x) using z-score normalization
        Ex = intensity.max(dim=2).values
        
        # Ex = intensity.max(dim=2).values  # (B, N)
        
        # new addition
        # peak = intensity.max(dim=2).values
        # mean = intensity.mean(dim=2)
        # energy = intensity.std(dim=2)
        
        # features = torch.stack(
        # [
        #     peak,
        #     mean,
        #     energy
        # ], dim=-1)
        
        # Ex = self.localization_head(features).squeeze(-1)

        # Ex_graph = Ex.unsqueeze(-1).unsqueeze(-1)  # (B,N,1,1)
        
        # Ex_graph = self.graph_refinement(
        #     Ex_graph,
        #     lap_per_feature[:, :1] if lap_per_feature.dim()==4 else lap_per_feature[:1]
        # )
        
        # Ex = Ex_graph.squeeze(-1).squeeze(-1)
        
        # Ex = (Ex - Ex.mean(dim=1, keepdim=True)) / (
        #     Ex.std(dim=1, keepdim=True) + 1e-8
        # )
        centroid = coords.mean(dim=1, keepdim=True)          # (B, 1, 2)
        coords_centered = coords - centroid                   # (B, N, 2)
        scale = coords_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)  # (B, 1, 1)
        coords_norm = coords_centered / scale                 # (B, N, 2)

        s_temp = torch.exp(self.log_temp_spatial) + 1e-4      
        s_temp = torch.clamp(s_temp, min=0.02, max=0.10)
        Ex_bounded = torch.clamp(Ex, max=5.0)       
        e_weights = torch.softmax(Ex_bounded / s_temp, dim=1)
        coords_pred_norm = (e_weights.unsqueeze(-1) * coords_norm).sum(dim=1)  # (B, 2)
        # coords_pred_e = (e_weights.unsqueeze(-1) * coords).sum(dim=1)
        coords_pred_e = coords_pred_norm * scale.squeeze(-1) + centroid.squeeze(1)
        # if coords.max() <= 1.0:
            # coords_pred_e = torch.clamp(coords_pred_e, min=0.0, max=1.0)
        
        # Frequency evidence
        P = (alpha.unsqueeze(0).unsqueeze(0)*band_energy_norm).clamp(min=1e-8)
        Q = (alpha.unsqueeze(0).unsqueeze(0)*neighbor_avg_norm).clamp(min=1e-8)
        M = (0.5 * (P + Q)).clamp(min=1e-8)
        
        KL_PM = (P * (P.log() - M.log())).sum(dim=-1)
        KL_QM = (Q * (Q.log() - M.log())).sum(dim=-1)
        raw_frequency_score = 0.5 * (KL_PM + KL_QM)
        freq_mean = raw_frequency_score.mean(dim=1, keepdim=True)
        freq_std = raw_frequency_score.std(dim=1, keepdim=True) + 1e-8
        frequency_score_norm = torch.sigmoid((raw_frequency_score - freq_mean) / freq_std)
        
        # Temporal confidence
        ex_mean = Ex.mean(dim=1, keepdim=True)
        ex_std  = Ex.std(dim=1, keepdim=True) + 1e-8
        temporal_score_norm = torch.sigmoid((Ex_bounded - ex_mean) / ex_std)
        
        # Neighborhood consistency
        raw_neighbor_difference = torch.abs(contrast.mean(dim=-1)) 
        neigh_mean = raw_neighbor_difference.mean(dim=1, keepdim=True)
        neigh_std = raw_neighbor_difference.std(dim=1, keepdim=True) + 1e-8
        neighbor_score_norm = torch.sigmoid((raw_neighbor_difference - neigh_mean) / neigh_std)
    

        px_features = torch.stack([
            frequency_score_norm.contiguous(), 
            temporal_score_norm.contiguous(), 
            neighbor_score_norm.contiguous()
        ], dim=-1)
        
        # (B, N)
        Px = self.px_head(px_features).squeeze(-1)
        
        # normalize so KL behaves well
        # Px = (Px - Px.mean(dim=1, keepdim=True)) / (Px.std(dim=1, keepdim=True) + 1e-8)
        
        Px = torch.softmax(Px / s_temp, dim=1)

        # return (
        #     coords_pred_e,
        #     time_pred,
        #     intensity,
        #     Px,
        #     raw_frequency_score,
        #     Ex,
        #     raw_neighbor_difference,
        #     band_energy_flat,
        #     alpha,
        # )
        
        return (
            coords_pred_e, time_pred, intensity, Px,
            frequency_score_norm,   
            temporal_score_norm,    
            neighbor_score_norm,    
            band_energy_flat, alpha,
        )

    def localize_by_frequency(self, Px, coords_batch, p_temperature=0.002):
        B, N, _ = coords_batch.shape
        
        # Normalize coordinates
        centroid = coords_batch.mean(dim=1, keepdim=True)        # (B, 1, 2)
        coords_c = coords_batch - centroid                        # (B, N, 2)
        scale    = coords_c.norm(dim=2).mean(dim=1)              # (B,)
        scale    = scale.clamp(min=1e-4)
        coords_n = coords_c / scale.view(B, 1, 1)               # (B, N, 2)
        
        # Soft-argmax over P(x) scores
        weights  = torch.softmax(Px / p_temperature, dim=1)      # (B, N)
        pred_norm = (weights.unsqueeze(-1) * coords_n).sum(dim=1) # (B, 2)
        
        # Denormalize -- explicit shapes prevent squeeze errors
        pred = pred_norm * scale.view(B, 1) + centroid.squeeze(1) # (B, 2)
        return pred
    
    def localize_by_entropy(self, Hx, coords, p_temperature=0.01):
        """
        H(x)-based localization using negative entropy (lower entropy means
        sharper signal and closer to source gets higher weight).
        """
        neg_Hx = -Hx
        mean = neg_Hx.mean(dim=1, keepdim=True)
        std = neg_Hx.std(dim=1, keepdim=True) + 1e-8
        normalized = (neg_Hx - mean) / std
        H_weights = torch.softmax(normalized / p_temperature, dim=1)
        
        return (H_weights.unsqueeze(-1) * coords).sum(dim=1)
    
    def get_explaination(self, Px, frequency_score, temporal_score, 
                         neighbor_score, band_energy, alpha):
       """
        Multi-level explanation for an event detection decision.
 
        Returns a dict with:
          spatial_explanation:       (B, N) normalized P(x) -- which
                                     sensors are closest to the source
          entropy:               (B, N) -- which sensors see sharp
                                     vs smooth signals
          learned_band_weights:      (n_bands,) -- the alpha_k values
                                     the model learned. Directly
                                     interpretable: which wavelet band
                                     is most informative for source
                                     proximity in this dataset.
          predicted_source_sensor:   (B,) -- index of sensor with
                                     highest P(x), i.e. predicted
                                     nearest to true event source.
        """ 
       
       return {
            "Px": Px,
            "frequency_score": frequency_score,
            "temporal_score": temporal_score,
            "neighbor_score": neighbor_score,
            "band_energy": band_energy,
            "wavelet_weights": alpha.detach(),
            "predicted_source_sensor": Px.argmax(dim=1),
        }
   
    def frequency_proximity_loss(self, Px, coords, true_location, sigma=0.1, p_temperature=0.01):
        """
        Auxiliary loss that explicitly trains P(x) to be highest at the
        sensor nearest the true event source.
     
        Without this, P(x) has NO gradient pushing it toward source
        proximity -- the main coordinate loss goes only through E(x).
     
        Px:            (B, N)
        coords:        (B, N, 2)
        true_location: (B, 2)
        sigma:         sharpness of the target distribution
     
        Returns scalar KL divergence loss.
        """
        centroid = coords.mean(dim=1, keepdim=True)
        coords_centered = coords - centroid
        scale = coords_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
        coords_norm = coords_centered / scale
        true_loc_norm = (true_location - centroid.squeeze(1)) / scale.squeeze(1)
        dist_to_source = torch.norm(coords_norm - true_loc_norm.unsqueeze(1), dim=2)
        # dist_to_source = torch.norm(coords - true_location.unsqueeze(1), dim=2)                                                            # (B, N)
        target = torch.softmax(-dist_to_source / sigma, dim=1)      # (B, N)
        
        mean = Px.mean(dim=1, keepdim=True)
        std = Px.std(dim=1, keepdim=True) + 1e-8
        normalized = (Px - mean) / std
        pred   = torch.softmax(normalized / p_temperature, dim=1)                           # (B, N)
        return F.kl_div(pred.clamp(min=1e-8).log(), target, reduction='batchmean')
    
    def localize_by_peak(self, intensity, coords):
        
        B, N, _ = intensity.shape
        centroid = coords.mean(dim=1, keepdim=True)
        coords_centered = coords - centroid
        scale = coords_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-4)
        coords_norm = coords_centered / scale

        peak_intensity = intensity.max(dim=2).values
        best_sensor = peak_intensity.argmax(dim=1)
        pred_norm = coords_norm[
            torch.arange(coords.shape[0], device=coords.device),
            best_sensor
        ]
        return pred_norm * scale.view(B, 1) + centroid.squeeze(1)

    def contrastive_intensity_loss(self, intensity, coords, true_location, margin=0.1, use_real_data=False):
      B, N, T = intensity.shape
      
      # 1. Use max intensity, not mean
      intensity_flat = intensity.max(dim=2).values  # (B, N) <- FIXED

      # 2. Compute Haversine distance for real data
      if use_real_data:
          from evaluate import Evaluator
          dist = torch.zeros(B, N, device=intensity.device)
          for b in range(B):
              for n in range(N):
                  dist[b, n] = Evaluator._compute_haversine_distance(
                      coords[b, n:n+1], true_location[b:b+1]
                  )
      else:
          dist = torch.norm(coords - true_location.unsqueeze(1), dim=2)

      # 3. Find closest (positive) and farthest (negative) sensors
      min_dist_idx = dist.argmin(dim=1)
      max_dist_idx = dist.argmax(dim=1)

      # 4. Contrastive loss
      pos_intensity = intensity_flat[torch.arange(B), min_dist_idx]
      neg_intensity = intensity_flat[torch.arange(B), max_dist_idx]
      loss = torch.relu(neg_intensity - pos_intensity + margin).mean()
    
      return loss

