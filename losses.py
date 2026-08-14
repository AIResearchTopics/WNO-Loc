# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:18:15 2026

@author: anjum
"""

# -*- coding: utf-8 -*-
"""
Optimized and Shape-Safe Localization Loss.
Fixes applied:
  - Inherited from nn.Module to natively align with standard PyTorch workflow architectures.
  - Added strict shape-matching assertions to prevent silent, dangerous tensor broadcasting bugs.
"""

import torch.nn as nn


class Localizationloss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def compute_loss(
            self,
            coords_pred,
            coords_true,
            time_pred,
            time_true
            ):
        """
        Computes the combined mean squared error for spatiotemporal localization.
        
        Shapes:
          coords_pred / coords_true: (B, 2)
          time_pred / time_true:     (B, 1)
        """
        # 1. Protect against silent broadcasting errors by enforcing exact shape alignment
        assert coords_pred.shape == coords_true.shape, \
            f"Coordinate shape mismatch! Pred: {coords_pred.shape}, True: {coords_true.shape}"
            
        assert time_pred.shape == time_true.shape, \
            f"Time shape mismatch! Pred: {time_pred.shape}, True: {time_true.shape}"

        # 2. Compute individual spatiotemporal loss components
        coord_loss = self.mse(coords_pred, coords_true)
        time_loss = self.mse(time_pred, time_true)
        
        # 3. Combine losses (You can multiply these by independent alpha/beta scalars 
        # later if your coordinate loss starts overpowering your time step loss)
        total_loss = coord_loss + time_loss
        
        return total_loss