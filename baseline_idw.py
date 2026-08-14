# -*- coding: utf-8 -*-
"""
Baseline 1: Inverse Distance Weighting (IDW) Geodesic Interpolator.
Features automated batch dimension handling and specific adversarial decoy logging profiles.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import torch

class IDWBaseline:
    def __init__(self, power=2.0, is_real_data=True):
        """
        Initializes the IDW baseline engine.
        """
        self.power = power
        self.is_real_data = is_real_data

    def _compute_distance_matrix(self, coords1, coords2):
        """
        Computes distances between a single point (coords1:) 
        and an array of grid points (coords2: [N, 2]).
        """
        if self.is_real_data:
            R = 6371.0
            lat1, lon1 = torch.deg2rad(coords1[..., 0]), torch.deg2rad(coords1[..., 1])
            lat2, lon2 = torch.deg2rad(coords2[..., 0]), torch.deg2rad(coords2[..., 1])
            
            dlat = lat2 - lat1.unsqueeze(-1)
            dlon = lon2 - lon1.unsqueeze(-1)
            
            a = torch.sin(dlat / 2)**2 + torch.cos(lat1.unsqueeze(-1)) * torch.sin(dlon / 2)**2 * torch.cos(lat2)
            c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
            return R * c
        else:
            # Pure, stable vector broadcasting subtraction math for flat simulations
            return torch.norm(coords2 - coords1.unsqueeze(0), dim=-1)

    def forward(self, U, coords, is_real_data=True):
        """
        Executes IDW tracking over incoming data batches.
        """
        self.is_real_data = is_real_data
        B, N, T, F = U.shape
        device = U.device
        
        # Collapse features via maximum channel variance
        intensity_field = U.max(dim=-1).values.contiguous() 
        max_peaks = intensity_field.max(dim=2).values       
        
        coords_pred_list = []
        
        # Loop explicitly over batch dimension B
        for b in range(B):
            b_coords = coords[b]   
            b_peaks = max_peaks[b] 
            
            max_sensor_idx = b_peaks.argmax()
            peak_station_coords = b_coords[max_sensor_idx] 
            
            distances = self._compute_distance_matrix(peak_station_coords, b_coords)
            
            weights = 1.0 / (distances + 1e-6)**self.power
            weights = weights * b_peaks 
            weights = weights / (weights.sum() + 1e-8) 
            
            pred_xy = (weights.unsqueeze(-1) * b_coords).sum(dim=0) 
            coords_pred_list.append(pred_xy)
            
        coords_pred_e = torch.stack(coords_pred_list).to(device)
        return coords_pred_e, intensity_field


def run_idw_evaluation(loader, label, is_real_data=True, power=2.0, logger=None, event_idx=0):
    """
    Evaluates the IDW model and persists structural scalar logs to disk.
    Automatically handles normal validation runs or adversarial decoy streams seamlessly.
    """
    idw_bench = IDWBaseline(power=power, is_real_data=is_real_data)
    idw_loc_errors = []
    dist_unit = "km" if is_real_data else "units"
    
    with torch.no_grad():
        for U, coords_batch, _, _, location, _ in loader:
            pred_coords, _ = idw_bench.forward(U, coords_batch, is_real_data=is_real_data)
            pred_coords = pred_coords.to(location.device)
            
            if is_real_data:
                from evaluate import Evaluator
                error = Evaluator._compute_haversine_distance(pred_coords, location)
            else:
                error = torch.norm(pred_coords - location, dim=1)
                
            idw_loc_errors.append(error)
            
    all_errors = torch.cat(idw_loc_errors)
    mean_err = all_errors.mean().item()
    std_err = all_errors.std().item()
    median_err = all_errors.median().item()
    
    # Check if this function call is executing a decoy test pass or a normal validation pass
    is_decoy_run = "decoy" in label.lower() or "adversarial" in label.lower()
    suffix = "Adversarial_Decoy_Baseline" if is_decoy_run else "IDW_Baseline"
    
    print(f"[{label} - {suffix.replace('_', ' ')}]")
    print(f"• IDW Localization Error E(x): {mean_err:.4f} {dist_unit} (std {std_err:.4f})")
    
    results_dict = {
        "label": f"{label}_{suffix}",
        "e_loc_mean": mean_err,
        "e_loc_std": std_err,
        "e_loc_median": median_err, 
        "e_loc_normalized": 0.0,
        "e_time_mean": 0.0,
        "e_time_std": 0.0,
        "e_time_p90": 0.0,
        "p_loc_mean": mean_err,
        "p_loc_std": std_err,
        "p_loc_median": median_err,  
        "peak_loc_mean":   0.0,
        "peak_loc_std":    0.0,
        "peak_loc_median": 0.0,
        "p_loc_normalized": 0.0,           
        "e_sensor_acc": 0.0,
        "p_top1_acc": 0.0,
        "p_proximity_acc": 0.0,
        "explainability_fidelity_delta": 0.0,
        "p_top3_acc": 0.0,
        "p_top5_acc": 0.0,
        "p_top10_acc": 0.0,
        "sensor_dist_mean": mean_err,
        "sensor_dist_std": std_err,
        "sensor_idx_error": 0.0,
        "freq_score_mean": 0.0,
        "freq_score_std": 0.0,
        "freq_score_max": 0.0,
        "temp_score_mean": 0.0,
        "temp_score_std": 0.0,
        "temp_score_max": 0.0,
        "neigh_score_mean": 0.0,
        "neigh_score_std": 0.0,
        "neigh_score_max": 0.0,
        "target_event_idx": event_idx,
        f"event_{event_idx}_true_sensor": 0,
        f"event_{event_idx}_pred_sensor": 0,
        f"event_{event_idx}_wavelet_weights": [0.0],
        f"event_{event_idx}_freq_raw": [0.0],
        f"event_{event_idx}_temp_raw": [0.0],
        f"event_{event_idx}_neigh_raw": [0.0]
    }
    
    if logger is not None:
        logger.write_txt_report(results_dict)
        logger.write_csv_metrics(results_dict)
        
    return mean_err
