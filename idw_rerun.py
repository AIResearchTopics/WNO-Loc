# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:04:43 2026

@author: anjum
"""
import os
import sys
import matplotlib
import config
_is_colab = 'google.colab' in sys.modules
_is_spyder = 'spyder' in sys.modules
_is_automated = os.environ.get("EXP_RUN_ID") is not None
if _is_spyder:
    pass
elif _is_colab or _is_automated or config._args.no_display:
    matplotlib.use('Agg')
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from synthetic_data import SyntheticEventGenerator
from dataset import EventDataset
from losses import Localizationloss
from model import LocalizationModel
from visualizations import (plot_sensor_map, plot_sensor_graph, plot_sensor_timeseries, plot_event_field)
from train import Trainer
from evaluate import Evaluator
from evaluationlogger import EvaluationLogger
import matplotlib.pyplot as plt
from real_dataset_loader import (scan_and_process_folder,build_fold,regions_to_dataset)

def get_nearest_sensor(coords_batch, true_location, is_real_data=True):
    """
    For each event in the batch, find which sensor is physically
    nearest to the true event source location.

    coords_batch:  (B, N, 2)
    true_location: (B, 2)
    is_real_data:  if True, use Haversine distance (km)
                   if False, use Euclidean distance (unit square)

    Returns: (B,) indices of the nearest sensor per event
    """
    if is_real_data:
        # Haversine distance in km
        R = 6371.0
        lat1 = torch.deg2rad(true_location[:, 0])          # (B,)
        lon1 = torch.deg2rad(true_location[:, 1])          # (B,)
        lat2 = torch.deg2rad(coords_batch[:, :, 0])        # (B, N)
        lon2 = torch.deg2rad(coords_batch[:, :, 1])        # (B, N)

        dlat = lat2 - lat1.unsqueeze(1)                    # (B, N)
        dlon = lon2 - lon1.unsqueeze(1)                    # (B, N)

        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1.unsqueeze(1))
             * torch.cos(lat2)
             * torch.sin(dlon / 2) ** 2)
        dist = 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))  # (B, N) in km
    else:
        dist = torch.norm(
            coords_batch - true_location.unsqueeze(1), dim=2
        )  # (B, N) in unit-square units

    return dist.argmin(dim=1)  # (B,)

plot_dir = os.path.join(config.OUTPUT_DIR, "plots")
os.makedirs(plot_dir, exist_ok=True)

SEED = config.SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
device = config.DEVICE
use_real_data = config.USE_REAL_DATA
N_SENSORS = config.N_SENSORS
T = config.T
DATA_SELECTOR = config.DATA_SELECTOR
pin = device.type in ['cuda', 'xpu']

config.print_config()

if use_real_data:

    all_regions = scan_and_process_folder(
        data_folder          = config.DATAFOLDER,
        window_length        = config.T,
        pre_event_hours      = 48,
        max_tier             = 3,
        exclude_diffuse      = False,
        max_wildfire_dist_km = 40.0,
        force_reprocess      = config.FORCE_REPROCESS,
    )

    N_FEATURES = all_regions[0]['n_features']

    train_regions, val_regions, test_regions, current_labels = build_fold(
        all_regions, fold=DATA_SELECTOR
    )

    train_set = regions_to_dataset(
        train_regions, k_neighbors=4, graph_sigma=0.2,
        coverage_prob=0.8, mask_seed=config.SEED
    )
    print(">>> train_set built, starting DataLoader") 
    val_set = regions_to_dataset(
        val_regions, k_neighbors=4, graph_sigma=0.2,
        coverage_prob=0.8, mask_seed=config.SEED
    )
    print(">>> val_set built")
    test_set = regions_to_dataset(
        test_regions, k_neighbors=4, graph_sigma=0.2,
        coverage_prob=0.8, mask_seed=config.SEED
    )
    print(">>> test_set built, starting training")
    if train_set is None:
        raise RuntimeError(
            f"Train set empty for fold '{DATA_SELECTOR}'. "
            f"Check city names and years in data folder."
        )

    print(f"\nFold '{DATA_SELECTOR}' ready:")
    print(f"  Train: {len(train_set):4d} events")
    print(f"  Val:   {len(val_set) if val_set else 0:4d} events")
    print(f"  Test:  {len(test_set) if test_set else 0:4d} events")

    batch_size = 1
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  pin_memory=pin
    )
    val_loader = DataLoader(
        val_set,   batch_size=batch_size, shuffle=False, pin_memory=pin
    ) if val_set else None
    test_loader = DataLoader(
        test_set,  batch_size=batch_size, shuffle=False, pin_memory=pin
    ) if test_set else None

    # Sanity-check visualizations
    sanity_ds = regions_to_dataset(
        [(val_regions or train_regions)[0]],
        mask_seed=config.SEED
    )
    U0, coords0, lap0, adj0, loc0, t0 = sanity_ds[0]
   
else:

    # ---------------------------------------------------------------------------
    # Generator
    # ---------------------------------------------------------------------------
    
    N_FEATURES = config.N_FEATURES
    
    generator = SyntheticEventGenerator(
        n_sensors=N_SENSORS, n_timesteps=T, n_features=N_FEATURES, random_seed=SEED
    )
    coords = generator.generate_sensor_locations()
    
    centre = coords.mean(axis=0)
    dist_to_centre = np.linalg.norm(coords - centre, axis=1)
    decoy_idx = int(np.argmax(dist_to_centre))
    decoy_loc = coords[decoy_idx]

    # ---------------------------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------------------------
    print("Building dataset (normal events):")
    normal_dataset = EventDataset(
        generator, num_events=500, coords=coords, lazy=False,
        k_neighbors=4, graph_sigma=0.2, coverage_prob=0.6, mask_seed=SEED,
        velocity=[0.1, 0.02, 0.05], broadening=[0.5, 4.0, 2.0],
    )
    
    print("\nBuilding dataset (adversarial events):")
    adversarial_dataset = EventDataset(
        generator, num_events=200, coords=coords, lazy=False,
        k_neighbors=4, graph_sigma=0.2, coverage_prob=0.6, mask_seed=SEED,
        velocity=[0.1, 0.02, 0.05], broadening=[0.5, 4.0, 2.0],
        decoy_location=decoy_loc, decoy_boost=1.5, decoy_sigma=0.15,
    )
    
    # Split normal data into pure training and pure validation
    n_val = int(0.2 * len(normal_dataset))
    n_train = len(normal_dataset) - n_val
    train_normal_set, val_set = random_split(
        normal_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Split adversarial data into pure training and pure validation
    n_adv_val = int(0.2 * len(adversarial_dataset))
    n_train_adv = len(adversarial_dataset) - n_adv_val
    train_adv_set, test_adv_set = random_split(
        adversarial_dataset, [n_train_adv, n_adv_val], generator=torch.Generator().manual_seed(SEED)
    )
    
    train_set = torch.utils.data.ConcatDataset([train_normal_set, train_adv_set])
    # train_set = torch.utils.data.ConcatDataset([train_normal_set])
    
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, pin_memory=pin)
    adv_loader   = DataLoader(test_adv_set, batch_size=8, shuffle=False, pin_memory=pin)
    
    print("\n--- Training Dataset Mixed Successfully ---")
    print(f"Total mixed training samples: {len(train_set)} ({n_train} Normal + {n_train_adv} Adversarial)")
    print(f"Clean validation samples: {len(val_set)}")
    print(f"Hidden adversarial validation samples: {len(test_adv_set)}")

    # ---------------------------------------------------------------------------
    # Sanity-check visualizations 
    # ---------------------------------------------------------------------------
    U0, coords0, lap0, adj0, loc0, t0 = normal_dataset[0]

    print("=== Loader diagnostic ===")
    print(f"val_loader:  {len(val_loader.dataset)} events, batch_size={val_loader.batch_size}")
    print(f"adv_loader:  {len(adv_loader.dataset)} events, batch_size={adv_loader.batch_size}")
    
    # If these datasets carry event_kwargs / generator params, print them too
    val_ds = val_loader.dataset
    adv_ds = adv_loader.dataset
    
    # If val_loader.dataset is a Subset (from random_split), unwrap it
    if hasattr(val_ds, 'dataset'):
        val_ds = val_ds.dataset
    if hasattr(adv_ds, 'dataset'):
        adv_ds = adv_ds.dataset
    
    print(f"val dataset event_kwargs:  {getattr(val_ds, 'event_kwargs', 'N/A')}")
    print(f"adv dataset event_kwargs:  {getattr(adv_ds, 'event_kwargs', 'N/A')}")
# =========================================================================== #
# RUN ENCAPSULATED IDW GEODESIC BASELINE
# =========================================================================== #
print("\n" + "="*80 + "\nRUNNING BASELINE 1: INVERSE DISTANCE WEIGHTING (IDW)\n" + "="*80)

logger = EvaluationLogger(output_dir=config.OUTPUT_DIR)
logger.log_config_snapshot(config)

# 1. Import the wrapper evaluator function directly
from baseline_idw import run_idw_evaluation

# 2. Trigger IDW with a single-line call matching your active dataset fold!
if use_real_data:
    run_idw_evaluation(val_loader, label=f"IDW_{current_labels['val']}", is_real_data=True, logger=logger)
    run_idw_evaluation(test_loader, label=f"IDW_{current_labels['test']}", is_real_data=True, logger=logger)
else:
    run_idw_evaluation(val_loader, label="IDW_Validation_Normal", is_real_data=False, logger=logger)
    run_idw_evaluation(adv_loader, label="IDW_Adversarial_Decoy_1.5x", is_real_data=False, logger=logger)