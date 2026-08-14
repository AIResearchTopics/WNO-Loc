# -*- coding: utf-8 -*-
"""
Advanced Evaluation Module for Spatiotemporal Localization Frameworks.
Natively integrates continuous E(x) and P(x) tracking errors to resolve redundancies.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import torch
import numpy as np

class Evaluator:
    def __init__(self, model, device, get_nearest_sensor_fn, print_results=True, logger=None, use_real_data=True):
        """
        Initializes the Evaluator class with required hardware allocations and state modules.
        """
        self.model = model
        self.device = device
        self.get_nearest_sensor = get_nearest_sensor_fn
        self.print_results = print_results
        self.logger = logger
        self.use_real_data = use_real_data
        
    @staticmethod
    def _compute_haversine_distance(coords1, coords2):
        """
        Computes great-circle Earth distances utilizing the Haversine formula.
        Expects shapes (B, 2) as [Latitude, Longitude]. Returns metrics in Kilometers (km).
        """
        R = 6371.0  # Earth's mean radius in km
        lat1, lon1 = torch.deg2rad(coords1[:, 0]), torch.deg2rad(coords1[:, 1])
        lat2, lon2 = torch.deg2rad(coords2[:, 0]), torch.deg2rad(coords2[:, 1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        return R * c
    
    @staticmethod
    def _compute_mean_spacing(coords_batch):
        """Compute mean nearest-neighbor sensor spacing in km."""
        B, N, _ = coords_batch.shape
        spacings = []
        for b in range(B):
            coords = coords_batch[b]  # (N, 2)
            R = 6371.0
            lat1 = torch.deg2rad(coords[:, 0]).unsqueeze(1)
            lon1 = torch.deg2rad(coords[:, 1]).unsqueeze(1)
            lat2 = torch.deg2rad(coords[:, 0]).unsqueeze(0)
            lon2 = torch.deg2rad(coords[:, 1]).unsqueeze(0)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (torch.sin(dlat/2)**2 
                 + torch.cos(lat1) * torch.cos(lat2) 
                 * torch.sin(dlon/2)**2)
            dist = 2 * R * torch.asin(torch.sqrt(a.clamp(0,1)))
            dist.fill_diagonal_(float('inf'))
            min_dist = dist.min(dim=1).values
            spacings.append(min_dist.mean())
        return torch.stack(spacings).mean()
    
    def nearest_k_accuracy(self, pred_sensor, coords_batch, location, k=3):
      B, N, _ = coords_batch.shape
      
      # Clamp k to network size -- prevents error on small networks (e.g. Munich: 5 sensors)
      k_eff = min(k, N)
      
      if self.use_real_data:
          R = 6371.0
          lat1 = torch.deg2rad(location[:, 0]).unsqueeze(1)
          lon1 = torch.deg2rad(location[:, 1]).unsqueeze(1)
          lat2 = torch.deg2rad(coords_batch[:, :, 0])
          lon2 = torch.deg2rad(coords_batch[:, :, 1])
          dlat = lat2 - lat1
          dlon = lon2 - lon1
          a = (torch.sin(dlat/2)**2
              + torch.cos(lat1) * torch.cos(lat2)
              * torch.sin(dlon/2)**2)
          dist = 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))
      else:
          dist = torch.norm(
              coords_batch - location.unsqueeze(1), dim=2
          )
      
      # Use k_eff instead of k
      k_nearest = dist.topk(k_eff, largest=False).indices  # (B, k_eff)
      correct = (pred_sensor.unsqueeze(1) == k_nearest).any(dim=1).float()
      return correct                                           # (B,)
    
    def evaluate(self, loader, label, event_idx=0, is_real_data=True):
        """
        Runs a comprehensive evaluation sweep over the selected data stream.
        Computes robust statistics, Top-K attributions, and structural ablativity deltas.
        """
        self.model.eval()
        
        # Spatial Coordinate Trackers for E(x) [Intensity Head]
        spatial_errors = []
        normalized_errors = []
        
        # FIXED: Added the missing spatial tracking array buffers for P(x) [Frequency Head]!
        p_spatial_errors = []
        p_normalized_errors = []
        
        # Temporal Trackers
        temporal_errors = []
        
        # Discrete Attribution & Proximity Accumulators
        p_sensor_correct = []
        p_top3_correct = []
        p_top5_correct = []
        p_top10_correct = []
        p_proximity_correct = []
        sensor_distance_error = []
        sensor_index_error = []
        peak_errors = []
        
        # Explainability Verification Tracking
        fidelity_deltas = []

        # Local caches for printing specific stats safely near the bottom
        frequency_score_last = torch.tensor(0.0)
        temporal_score_last = torch.tensor(0.0)
        neighbor_score_last = torch.tensor(0.0)

        with torch.no_grad():
            for U, coords_batch, lap_batch, adj_batch, location, time_batch in loader:
                U = U.to(self.device, non_blocking=True)
                coords_batch = coords_batch.to(self.device, non_blocking=True)
                location = location.to(self.device, non_blocking=True)
                time_batch = time_batch.to(self.device, non_blocking=True)
                lap_batch = lap_batch.to(self.device)
                adj_batch = adj_batch.to(self.device)

                # 1. Forward Pass Execution
                coords_pred_e, t_pred, intensity, Px, \
                frequency_score, temporal_score, neighbor_score, \
                band_energy, alpha = self.model(U, coords_batch, lap_batch, adj_batch)
                
                # Retrieve the frequency-projected coordinates P(x) cleanly
                coords_pred_p = self.model.localize_by_frequency(Px, coords_batch, p_temperature=0.002)
                explanations = self.model.get_explaination(Px, frequency_score, temporal_score, neighbor_score, band_energy, alpha)

                # Force shape layout invariance across both continuous tracking vectors
                if coords_pred_e.shape != location.shape:
                    coords_pred_e = coords_pred_e.reshape(location.shape)
                if coords_pred_p.shape != location.shape:
                    coords_pred_p = coords_pred_p.reshape(location.shape)
                coords_pred_peak = self.model.localize_by_peak(intensity, coords_batch)
                if coords_pred_peak.shape != location.shape:
                    coords_pred_peak = coords_pred_peak.reshape(location.shape)    
                # Cache values for post-loop printing signatures safely
                frequency_score_last = frequency_score
                temporal_score_last = temporal_score
                neighbor_score_last = neighbor_score

                # 2. Compute Absolute Spatial Trajectory Tracking Errors for E(x) and P(x)
                if is_real_data:
                    errors_e = self._compute_haversine_distance(coords_pred_e, location)
                    errors_p = self._compute_haversine_distance(coords_pred_p, location)
                else:
                    errors_e = torch.norm(coords_pred_e - location, dim=1)
                    errors_p = torch.norm(coords_pred_p - location, dim=1)
                
                if is_real_data:
                    errors_peak = self._compute_haversine_distance(coords_pred_peak, location)
                else:
                    errors_peak = torch.norm(coords_pred_peak - location, dim=1)
                peak_errors.append(errors_peak)
                
                spatial_errors.append(errors_e)
                p_spatial_errors.append(errors_p) # <-- FIXED: Stacking P(x) tracking values!

                # 3. Compute Bounded Spatial Coordinate Error Normalized by Bounding Diagonal
                for b_idx in range(coords_batch.size(0)):
                    if is_real_data:
                        grid = coords_batch[b_idx]
                        min_bounds = torch.stack([grid[:, 0].min(), grid[:, 1].min()])
                        max_bounds = torch.stack([grid[:, 0].max(), grid[:, 1].max()])
                        grid_diagonal = self._compute_haversine_distance(min_bounds.unsqueeze(0), max_bounds.unsqueeze(0)).item()
                    else:
                        grid_diagonal = 1.0
                        
                    norm_err_e = errors_e[b_idx].item() / (grid_diagonal + 1e-8)
                    norm_err_p = errors_p[b_idx].item() / (grid_diagonal + 1e-8)
                    
                    normalized_errors.append(norm_err_e)
                    p_normalized_errors.append(norm_err_p)

                # 4. Compute Temporal Ambiguity
                temporal_errors.append(torch.abs(t_pred.squeeze(-1) - time_batch.squeeze(-1)))

                # 5. Discrete Target Attribution Counters (Top-1, Top-3, Top-5)
                pred_sensor = explanations["predicted_source_sensor"]
                source_sensor = self.get_nearest_sensor(coords_batch, location, self.use_real_data)
                
                # p_sensor_correct.append((pred_sensor == source_sensor).float())
                
                # top3 = Px.topk(3, dim=1).indices
                # top5 = Px.topk(5, dim=1).indices
                # p_top3_correct.append((top3 == source_sensor.unsqueeze(-1)).any(dim=-1).float())
                # p_top5_correct.append((top5 == source_sensor.unsqueeze(-1)).any(dim=-1).float())

                nk1 = self.nearest_k_accuracy(pred_sensor, coords_batch, location, k=1)
                nk3 = self.nearest_k_accuracy(pred_sensor, coords_batch, location, k=3)
                nk5 = self.nearest_k_accuracy(pred_sensor, coords_batch, location, k=5)
                nk10 = self.nearest_k_accuracy(pred_sensor, coords_batch, location, k=10)

                p_sensor_correct.append(nk1)
                p_top3_correct.append(nk3)
                p_top5_correct.append(nk5)
                p_top10_correct.append(nk10)

                # 6. Physical Proximity Clustered Metrics
                batch_indices = torch.arange(coords_batch.size(0), device=self.device)
                true_sensor_coords = coords_batch[batch_indices, source_sensor]
                pred_sensor_coords = coords_batch[batch_indices, pred_sensor]
                
                if is_real_data:
                    spatial_dist = self._compute_haversine_distance(pred_sensor_coords, true_sensor_coords)
                    mean_spacing = self._compute_mean_spacing(coords_batch)
                    threshold = torch.clamp(mean_spacing * 2.0, min=15.0, max=50.0)
                    proximity_match = (spatial_dist <= threshold).float()
                else:
                    spatial_dist = torch.norm(pred_sensor_coords - true_sensor_coords, dim=1)
                    proximity_match = (spatial_dist <= 0.15).float()
                    
                sensor_distance_error.append(spatial_dist)
                sensor_index_error.append((pred_sensor - source_sensor).abs().float())
                p_proximity_correct.append(proximity_match)

                # 7. EXPLAINABILITY DEGRADATION VIA NEIGHBORHOOD ABLATION TEST
                U_perturbed = U.clone()
                for b_idx in range(coords_batch.size(0)):
                    top_node = pred_sensor[b_idx]
                    U_perturbed[b_idx, top_node, :, :] = 0.0
                
                coords_pred_perturbed, _, _, _, _, _, _, _, _ = self.model(U_perturbed, coords_batch, lap_batch, adj_batch)
                
                if coords_pred_perturbed.shape != location.shape:
                    coords_pred_perturbed = coords_pred_perturbed.reshape(location.shape)

                if is_real_data:
                    perturbed_dist = self._compute_haversine_distance(coords_pred_perturbed, location)
                else:
                    perturbed_dist = torch.norm(coords_pred_perturbed - location, dim=1)
                
                fidelity_deltas.append(perturbed_dist - errors_e)

        # ================================================================= #
        # Robust Array Reductions & Percentile Boundary Formulations
        # ================================================================= #
        all_spatial = torch.cat(spatial_errors).cpu().numpy()
        all_p_spatial = torch.cat(p_spatial_errors).cpu().numpy() # <-- FIXED REDUCTION
        all_temporal = torch.cat(temporal_errors).cpu().numpy()
        all_fidelity = torch.cat(fidelity_deltas).cpu().numpy()
        
        all_peak = torch.cat(peak_errors)
        mean_peak   = all_peak.mean().item()
        std_peak    = all_peak.std().item()
        median_peak = all_peak.median().item()
        
        mean_spatial = np.mean(all_spatial)
        median_spatial = np.median(all_spatial)
        std_spatial = np.std(all_spatial)
        mean_norm_err = np.mean(normalized_errors)
        
        # FIXED: Extract statistical benchmarks for your Frequency Localization head P(x)!
        mean_p_spatial = np.mean(all_p_spatial)
        median_p_spatial = np.median(all_p_spatial)
        std_p_spatial = np.std(all_p_spatial)
        mean_p_norm_err = np.mean(p_normalized_errors)
        
        mean_temporal = np.mean(all_temporal)
        std_temporal = np.std(all_temporal)
        p90_temporal = np.percentile(all_temporal, 90)
        
        top1_acc = torch.cat(p_sensor_correct).mean().item() * 100
        top3_acc = torch.cat(p_top3_correct).mean().item() * 100
        top5_acc = torch.cat(p_top5_correct).mean().item() * 100
        top10_acc = torch.cat(p_top10_correct).mean().item() * 100
        prox_acc = torch.cat(p_proximity_correct).mean().item() * 100
        
        mean_fidelity = np.mean(all_fidelity)
        sensor_dist = torch.cat(sensor_distance_error)
        sensor_error = torch.cat(sensor_index_error)
        dist_unit = "km" if is_real_data else "units"
        prox_threshold = "15km" if is_real_data else "0.15 units"
        
        # ================================================================= #
        # Robust Visual Console Printing Layout Summary Logs
        # ================================================================= #
        if self.print_results:
            print("\n" + "="*80 + f"\nPERFORMANCE REPORT PROFILE: [{label}]\n" + "="*80)
            print(f"• E(x) Mean Localization Error   : {mean_spatial:.4f} {dist_unit} (std {std_spatial:.4f})")
            print(f"• E(x) Median Localization Error : {median_spatial:.4f} {dist_unit} [ROBUST STATE]")
            print(f"• E(x) Normalized Grid Error     : {mean_norm_err:.4f}")
            print(f"• P(x) Mean Localization Error   : {mean_p_spatial:.4f} {dist_unit} (std {std_p_spatial:.4f}) [FREQUENCY HEAD]")
            print(f"• P(x) Median Localization Error : {median_p_spatial:.4f} {dist_unit} [FREQUENCY HEAD]")
            print(f"• P(x) Normalized Grid Error     : {mean_p_norm_err:.4f}")
            print(f"• Mean Temporal Onset Error E(t) : {mean_temporal:.4f} hours (std {std_temporal:.4f})")
            print(f"• 90th Percentile Time Error     : {p90_temporal:.4f} hours [BOUNDED TAIL]")
            print("-"*80)
            print(f"• Peak Sensor Loc Error          : {mean_peak:.4f} (std {std_peak:.4f})")
            print("-"*80)
            print(f"• P(x) Source Top-1 Accuracy    : {top1_acc:.2f}%")
            print(f"• P(x) Source Top-3 Accuracy    : {top3_acc:.2f}%")
            print(f"• P(x) Source Top-5 Accuracy    : {top5_acc:.2f}%")
            print(f"• P(x) Source Top-10 Accuracy   : {top10_acc:.2f}%")
            print(f"• P(x) Proximity Accuracy       : {prox_acc:.2f}% (Tolerance Window <= {prox_threshold})")
            print(f"• Explainability Fidelity Δ     : {mean_fidelity:.4f} {dist_unit} tracking degradation")
            print(f"• Continuous Sensor-to-Sensor   : {sensor_dist.mean().item():.4f} {dist_unit}")
            print("="*80)
        
            print("\nExplainability statistics")
            print(f"Frequency score (Wavelet Space) : mean={frequency_score_last.mean().item():.4f}, std={frequency_score_last.std().item():.4f}, max={frequency_score_last.max().item():.4f}")
            print(f"Temporal score (Onset Field)    : mean={temporal_score_last.mean().item():.4f}, std={temporal_score_last.std().item():.4f}, max={temporal_score_last.max().item():.4f}")
            print(f"Neighbor score (Topology Ring)  : mean={neighbor_score_last.mean().item():.4f}, std={neighbor_score_last.std().item():.4f}, max={neighbor_score_last.max().item():.4f}")
            print(f"Sensor Index Error              : {sensor_error.mean().item():.4f}")
        
            print(f"\nEvent {event_idx} Target Snapshot Diagnostics")
            print("True source sensor        :", source_sensor[event_idx].item())
            print("Predicted source sensor   :", explanations["predicted_source_sensor"][event_idx].item())
            
            alpha_np = explanations["wavelet_weights"].cpu().numpy()
            if alpha_np.ndim == 0:
                print("Wavelet weights           :", float(alpha_np))
            else:
                print("Wavelet weights           :", alpha_np.tolist())
        
        # ================================================================= #
        # Tabular Payload Generation For Disk Logger Modules
        # ================================================================= #
        alpha_np = explanations["wavelet_weights"].cpu().numpy()
        
        # FIXED: Dynamic string keys now feature structured underscores 
        # matching your EvaluationLogger column schema perfectly 1-to-1!
        results_dict = {
            "label": label,
            "e_loc_mean": float(mean_spatial),
            "e_loc_std": float(std_spatial),
            "e_loc_median": float(median_spatial),
            "e_loc_normalized": float(mean_norm_err),
            "p_loc_mean": float(mean_p_spatial), 
            "p_loc_std": float(std_p_spatial), 
            "p_loc_median": float(median_p_spatial), 
            "p_loc_normalized": float(mean_p_norm_err),
            "peak_loc_mean":   float(mean_peak),
            "peak_loc_std":    float(std_peak),
            "peak_loc_median": float(median_peak),
            "e_time_mean": float(mean_temporal),
            "e_time_std": float(std_temporal),
            "e_time_p90": float(p90_temporal),
            "p_top1_acc": float(top1_acc),
            "p_top3_acc": float(top3_acc),
            "p_top5_acc": float(top5_acc),
            "p_top10_acc": float(top10_acc),
            "p_proximity_acc": float(prox_acc),
            "explainability_fidelity_delta": float(mean_fidelity),
            "sensor_dist_mean": float(sensor_dist.mean().item()),
            "sensor_dist_std": float(sensor_dist.std().item()),
            "sensor_idx_error": float(sensor_error.mean().item()),
            "freq_score_mean": float(frequency_score_last.mean().item()),
            "freq_score_std": float(frequency_score_last.std().item()),
            "freq_score_max": float(frequency_score_last.max().item()),
            "temp_score_mean": float(temporal_score_last.mean().item()),
            "temp_score_std": float(temporal_score_last.std().item()),
            "temp_score_max": float(temporal_score_last.max().item()),
            "neigh_score_mean": float(neighbor_score_last.mean().item()),
            "neigh_score_std": float(neighbor_score_last.std().item()),
            "neigh_score_max": float(neighbor_score_last.max().item()),
            "target_event_idx": event_idx,
            f"event_{event_idx}_true_sensor": int(source_sensor[event_idx].item()),
            f"event_{event_idx}_pred_sensor": int(explanations["predicted_source_sensor"][event_idx].item()),
            f"event_{event_idx}_wavelet_weights": alpha_np.tolist() if alpha_np.ndim > 0 else [float(alpha_np)]
        }
        
        if self.logger is not None:
            self.logger.write_txt_report(results_dict)
            self.logger.write_csv_metrics(results_dict)
        
        return all_spatial, all_temporal, top1_acc, prox_acc, mean_fidelity

def run_causal_curves(model, loader, device, is_real_data=True, steps=5):
    """
    Computes Deletion and Insertion trajectory error curves based on P(x) attention rankings.
    Measures causal necessity and sufficiency across the monitoring network.
    """
    model.eval()
    
    # Pre-allocate dictionary arrays to accumulate step-by-step errors
    deletion_history = {step: [] for step in range(steps + 1)}
    insertion_history = {step: [] for step in range(steps + 1)}
    
    with torch.no_grad():
        for U, coords_batch, lap_batch, adj_batch, location, _ in loader:
            U = U.to(device, non_blocking=True)
            coords_batch = coords_batch.to(device, non_blocking=True)
            location = location.to(device, non_blocking=True)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            B, N, T, F = U.shape
            
            # 1. Run baseline forward pass to extract attention ranks (Px)
            coords_pred, _, _, Px, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)
            
            # Sort sensors per batch item from most important to least important
            top_ranked_nodes = Px.sort(dim=1, descending=True).indices # Shape: (B, N)
            
            # 2. RUN DELETION LOOP (Gradually blanking out top nodes)
            U_deletion = U.clone()
            for step in range(steps + 1):
                # Run prediction on current perturbed state
                pred, _, _, _, _, _, _, _, _ = model(U_deletion, coords_batch, lap_batch, adj_batch)
                if pred.shape != location.shape: pred = pred.reshape(location.shape)
                
                err = Evaluator._compute_haversine_distance(pred, location) if is_real_data else torch.norm(pred - location, dim=1)
                deletion_history[step].append(err.cpu())
                
                # Mute the next highest ranked node for the upcoming step
                if step < steps:
                    for b in range(B):
                        target_node = top_ranked_nodes[b, step]
                        U_deletion[b, target_node, :, :] = 0.0
                        
            # 3. RUN INSERTION LOOP (Starting from scratch and revealing top nodes)
            U_insertion = torch.zeros_like(U)
            for step in range(steps + 1):
                # Run prediction on current built state
                pred, _, _, _, _, _, _, _, _ = model(U_insertion, coords_batch, lap_batch, adj_batch)
                if pred.shape != location.shape: pred = pred.reshape(location.shape)
                
                err = Evaluator._compute_haversine_distance(pred, location) if is_real_data else torch.norm(pred - location, dim=1)
                insertion_history[step].append(err.cpu())
                
                # Reveal the next highest ranked node data matrix
                if step < steps:
                    for b in range(B):
                        target_node = top_ranked_nodes[b, step]
                        U_insertion[b, target_node, :, :] = U[b, target_node, :, :]

    # Condense all batch listings into a single final mean trajectory vector
    final_deletion_curve = [torch.cat(deletion_history[s]).mean().item() for s in range(steps + 1)]
    final_insertion_curve = [torch.cat(insertion_history[s]).mean().item() for s in range(steps + 1)]
    
    print("\n" + "="*80 + "\nCAUSAL EXPLAINABILITY CURVES COMPILATION COMPLETE\n" + "="*80)
    print(f"• Deletion Steps (0 to {steps} nodes removed) : {[round(x, 4) for x in final_deletion_curve]}")
    print(f"• Insertion Steps (0 to {steps} nodes revealed) : {[round(x, 4) for x in final_insertion_curve]}")
    print("="*80)
    
    return final_deletion_curve, final_insertion_curve

