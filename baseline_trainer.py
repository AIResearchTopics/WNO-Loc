# -*- coding: utf-8 -*-
"""
Decoupled Training & Evaluation Suite for Deep Learning Baselines.
Natively computes robust statistics (Medians, 90th Percentiles) for fair logging benchmarks.

Updated: BaselineTrainer.train() now tracks and returns the model state
with the best validation loss seen at any evaluated epoch (mirroring the
same best-checkpoint logic added to train.py's Trainer.train()), so every
model in the comparison -- WNO and all baselines -- is evaluated at its
own best checkpoint rather than whatever the final epoch happens to
leave, keeping the comparison symmetric in both directions.

Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import time
import torch
import numpy as np
import torch.nn.functional as F

class BaselineTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, train_loader, val_loader, device):
        """
        Initializes an independent training controller for baselines.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def run_epoch(self, loader, train=True):
        """
        Runs a single step-by-step optimization or validation pass.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        context = torch.enable_grad() if train else torch.no_grad()
        
        with context:
            for U, coords, lap, adj, location, time_batch in loader:
                U = U.to(self.device, non_blocking=True)
                coords = coords.to(self.device, non_blocking=True)
                location = location.to(self.device, non_blocking=True)
                time_batch = time_batch.to(self.device, non_blocking=True)
                lap = lap.to(self.device)
                adj = adj.to(self.device)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                pred_coords, pred_time, _, _, _, _, _, _, _ = self.model(U, coords, lap, adj)

                # Safe functional calculations bypass rigid custom nn.Module restrictions
                loss_loc = F.mse_loss(pred_coords, location)
                loss_time = F.mse_loss(pred_time.squeeze(-1), time_batch.squeeze(-1))
                loss = loss_loc + loss_time

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self, epochs, display_epoch=20):
        """
        Orchestrates complete independent baseline optimization loops.

        Also tracks the model state with the best validation loss seen
        at any evaluated epoch, and returns it alongside the final
        train_loss -- since the final epoch is not guaranteed to be the
        best one for this model, evaluating from the best checkpoint
        instead of the last epoch keeps the comparison fair against the
        main WNO model, which uses the same best-checkpoint selection.

        Returns
        -------
        train_loss : float
            The training loss from the final epoch (unchanged behavior).
        best_state : dict or None
            state_dict of the model at its best validation-loss epoch.
            None only if the loop somehow never evaluated (shouldn't
            happen given epoch 0 and the final epoch are always
            evaluated).
        """
        print(f"Beginning standalone baseline optimization path for {epochs} epochs...")

        best_val_loss = float('inf')
        best_state = None
        best_epoch = -1

        for epoch in range(epochs):
            t0 = time.time()
            train_loss = self.run_epoch(self.train_loader, train=True)
            epoch_time = time.time() - t0
            
            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % display_epoch == 0 or epoch == (epochs - 1):
                val_loss = self.run_epoch(self.val_loader, train=False)

                # NEW: checkpoint if this is the best validation loss so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

                print(f"Epoch {epoch:3d}/{epochs} | Baseline Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
            else:
                print(f"Epoch {epoch:3d}/{epochs} completed | Current Train Loss: {train_loss:.4f}", end="\r")

        print(f"\nBest Val Loss = {best_val_loss:.4f} at epoch {best_epoch} (final epoch was {epochs-1})")
        print("Baseline training execution completed successfully!")

        return train_loss, best_state


def run_baseline_evaluation(model, loader, label, device, is_real_data=True, logger=None, event_idx=0):
    """
    Independent Evaluation loop for Deep Learning Baselines (LSTM, FNO, Transformer).
    Fully synchronized to compute and log robust percentiles into your upgraded logger.

    Note: this function evaluates whatever weights are currently loaded
    into `model` -- callers are responsible for restoring the best
    checkpoint (via model.load_state_dict(best_state)) before calling
    this, if best-checkpoint selection is desired. This function itself
    doesn't change; only what state the model is in when it's called.
    """
    model.eval()
    e_loc_errors = []
    e_time_errors = []
    
    with torch.no_grad():
        for U, coords_batch, lap_batch, adj_batch, location, time_batch in loader:
            U = U.to(device, non_blocking=True)
            coords_batch = coords_batch.to(device, non_blocking=True)
            location = location.to(device, non_blocking=True)
            time_batch = time_batch.to(device, non_blocking=True)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)

            coords_pred_e, t_pred, _, _, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)

            if coords_pred_e.shape != location.shape:
                coords_pred_e = coords_pred_e.reshape(location.shape)

            if is_real_data:
                from evaluate import Evaluator
                e_loc_errors.append(Evaluator._compute_haversine_distance(coords_pred_e, location))
            else:
                e_loc_errors.append(torch.norm(coords_pred_e - location, dim=1))

            e_time_errors.append(torch.abs(t_pred.squeeze(-1) - time_batch.squeeze(-1)))

    # Convert error queues into robust NumPy reduction matrices
    all_spatial = torch.cat(e_loc_errors).cpu().numpy()
    all_temporal = torch.cat(e_time_errors).cpu().numpy()
    
    mean_loc = float(np.mean(all_spatial))
    median_loc = float(np.median(all_spatial))
    std_loc = float(np.std(all_spatial))
    
    mean_time = float(np.mean(all_temporal))
    std_time = float(np.std(all_temporal))
    p90_time = float(np.percentile(all_temporal, 90))

    dist_unit = "km" if is_real_data else "units"
    print(f"\n[{label} - Baseline Performance Summary]")
    print(f"• E(x) Mean Localization Error   : {mean_loc:.4f} {dist_unit} (std {std_loc:.4f})")
    print(f"• E(x) Median Localization Error : {median_loc:.4f} {dist_unit} [ROBUST STATE]")
    print(f"• E(t) Mean Temporal Onset Error : {mean_time:.4f} hours")
    print(f"• E(t) 90th Percentile Time Error : {p90_time:.4f} hours [BOUNDED TAIL]")

    # FIXED: Fully populated results dictionary passes all new percentile and median parameters
    # to eliminate any KeyError boundaries inside evaluationlogger.py!
    results_dict = {
        "label": label,
        "e_loc_mean": mean_loc,
        "e_loc_std": std_loc,
        "e_loc_median": median_loc,
        "e_loc_normalized": 0.0,
        "p_loc_mean": mean_loc,  # Fallback fields maintain continuous spreadsheet column layouts
        "p_loc_std": std_loc,
        "p_loc_median": median_loc,
        "peak_loc_mean":   0.0,
        "peak_loc_std":    0.0,
        "peak_loc_median": 0.0,
        "p_loc_normalized": 0.0,
        "e_time_mean": mean_time,
        "e_time_std": std_time,
        "e_time_p90": p90_time,
        "p_top1_acc": 0.0,
        "p_top3_acc": 0.0,
        "p_top5_acc": 0.0,
        "p_top10_acc": 0.0,
        "p_proximity_acc": 0.0,
        "explainability_fidelity_delta": 0.0,
        "sensor_dist_mean": mean_loc,
        "sensor_dist_std": std_loc,
        "sensor_idx_error": 0.0,
        "freq_score_mean": 0.0, "freq_score_std": 0.0, "freq_score_max": 0.0,
        "temp_score_mean": 0.0, "temp_score_std": 0.0, "temp_score_max": 0.0,
        "neigh_score_mean": 0.0, "neigh_score_std": 0.0, "neigh_score_max": 0.0,
        "target_event_idx": event_idx,
        f"event_{event_idx}_true_sensor": 0,
        f"event_{event_idx}_pred_sensor": 0,
        f"event_{event_idx}_wavelet_weights": [0.0]
    }

    if logger is not None:
        logger.write_txt_report(results_dict)
        logger.write_csv_metrics(results_dict)

    return mean_loc, mean_time