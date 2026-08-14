# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 13:14:13 2026
@author: anjum
"""

import torch
import torch.nn.functional as F
import time
import os
import config

class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        lambda_p,
        lambda_c=0.05,
        use_real_data=True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.use_real_data = use_real_data

    def run_epoch(self, loader, sigma=0.1, p_temperature=0.1, train=True):

        self.model.train(train)

        total_loss = torch.tensor(0.0, device=self.device)
        total_loc = torch.tensor(0.0, device=self.device)
        total_p = torch.tensor(0.0, device=self.device)

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:

            for (
                U,
                coords_batch,
                lap_batch,
                adj_batch,
                location,
                time_batch,
            ) in loader:

                U = U.to(self.device, non_blocking=True)
                coords_batch = coords_batch.to(self.device, non_blocking=True)
                location = location.to(self.device, non_blocking=True)
                time_batch = time_batch.to(self.device, non_blocking=True)
                lap_batch = lap_batch.to(self.device, non_blocking=True)
                adj_batch = adj_batch.to(self.device, non_blocking=True)

                (
                    coords_pred_e,
                    t_pred,
                    intensity,
                    Px,
                    frequency_score,
                    temporal_score,
                    neighbor_score,
                    band_energy,
                    alpha,
                ) = self.model(
                    U,
                    coords_batch,
                    lap_batch,
                    adj_batch,
                )

                #############################################
                # Localization loss
                #############################################

                loss_loc = self.loss_fn.compute_loss(
                    coords_pred_e,
                    location,
                    t_pred,
                    time_batch,
                )

                #############################################
                # Frequency proximity loss
                #############################################

                loss_p = self.model.frequency_proximity_loss(
                    Px,
                    coords_batch,
                    location,
                    sigma=sigma,
                    p_temperature=p_temperature,
                )

                #############################################
                # Consistency loss
                #############################################

                Ex = intensity.max(dim=2).values

                spatial_temp = (
                    torch.exp(self.model.log_temp_spatial).detach()
                    + 1e-4
                )

                Ex_prob = torch.softmax(
                    Ex / spatial_temp,
                    dim=1,
                )

                Px_prob = torch.softmax(
                    Px / spatial_temp,
                    dim=1,
                )

                loss_consistency = F.kl_div(
                    Px_prob.log(),
                    Ex_prob.detach(),
                    reduction="batchmean",
                )

                if self.use_real_data:
                    contrastive_loss = self.model.contrastive_intensity_loss(
                        intensity, coords_batch, location, use_real_data=self.use_real_data, margin=0.1, 
                    )
                    contrastive_weight = 0.1  # Adjust as needed
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    contrastive_weight = 0.0

                alpha_bands = alpha[1:]  # Exclude approximation band (band 0)
                alpha_entropy = -torch.sum(alpha_bands * torch.log(alpha_bands + 1e-8))

                # We want to penalize uniform weights (high entropy)
                # So we add a loss that encourages lower entropy
                # Higher when weights are uniform

                # Add to total loss (ONLY for real data)
                if self.use_real_data:
                    entropy_weight = 0.01
                else:
                    entropy_weight = 0.0

                loss = (
                    loss_loc 
                    + self.lambda_p * loss_p 
                    + self.lambda_c * loss_consistency 
                    + contrastive_weight * contrastive_loss
                    + entropy_weight * alpha_entropy  
                )

                #############################################
                # Total loss
                #############################################

                if train:

                    self.optimizer.zero_grad(set_to_none=True)

                    loss.backward()

                    self.optimizer.step()

                total_loss += loss * U.size(0)
                total_loc += loss_loc * U.size(0)
                total_p += loss_p * U.size(0)

        if train:
            self.scheduler.step()

        return (
            (total_loss / len(loader.dataset)).item(),
            (total_loc / len(loader.dataset)).item(),
            (total_p / len(loader.dataset)).item(),
        )

    def train(self, epochs, display_epoch=20):

        training_history = []

        best_val_loc = float("inf")
        best_state = None
        best_epoch = -1

        for epoch in range(epochs):

            t0 = time.time()

            train_loss, loss_loc, loss_p = self.run_epoch(
                self.train_loader,
                train=True,
            )

            epoch_time = time.time() - t0

            if epoch % display_epoch == 0 or epoch == epochs - 1:

                val_loss, val_loc, val_p = self.run_epoch(
                    self.val_loader,
                    train=False,
                )

                if val_loc < best_val_loc:
                    best_val_loc = val_loc
                    best_epoch   = epoch
                    best_state   = {
                        k: v.detach().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    # Save checkpoint to disk immediately
                    checkpoint_path = os.path.join(
                        config.OUTPUT_DIR, 'best_checkpoint.pt'
                    )
                    torch.save({
                        'epoch':      epoch,
                        'model_state_dict': best_state,
                        'val_loc':    val_loc,
                        'val_loss':   val_loss,
                    }, checkpoint_path)

                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: {train_loss:.4f} "
                    f"(Loc {loss_loc:.4f}, P {loss_p:.4f}) | "
                    f"Val: {val_loss:.4f} "
                    f"(Loc {val_loc:.4f}, P {val_p:.4f}) | "
                    f"{epoch_time:.2f}s"
                )

            else:

                val_loss = None
                val_loc = None
                val_p = None

                print(
                    f"Epoch {epoch:3d}/{epochs} "
                    f"Train Loss {train_loss:.4f}",
                    end="\r",
                )

            training_history.append({

                "total": train_loss,
                "loc": loss_loc,
                "p": loss_p,

                "val_total": val_loss,
                "val_loc": val_loc,
                "val_p": val_p,

            })

        print(
            f"\nBest validation localization loss = "
            f"{best_val_loc:.4f} "
            f"at epoch {best_epoch}"
        )

        return training_history, best_state
