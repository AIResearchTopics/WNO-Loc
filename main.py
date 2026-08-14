# -*- coding: utf-8 -*-
"""
Training and evaluation with the real WNO-based localization model.

Two evaluation modes:
  - normal: standard single-source events (no decoy)
  - adversarial: decoy signal placed at a far sensor, decoy_boost=1.5,
    to test whether E(x) gets fooled and P(x) resists

@author: usman.anjum
"""

"""
Optimized Pipeline Runner for Spatiotemporal Localization Framework.
Optimizations applied:
  - Enabled native host memory pinning (pin_memory=True) for Intel XPU architectures.
  - Eliminated per-batch blocking CPU-GPU synchronization loops (loss.item() bottleneck).
  - Ensured graph visualization array extraction is explicitly device-agnostic.
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

    plot_sensor_map(
        coords0.detach().cpu().numpy(),
        source_location=loc0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_sensor_map.png")
    )
    plot_sensor_graph(
        coords0.detach().cpu().numpy(),
        adj0[1].detach().cpu().numpy(),
        source_location=loc0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_sensor_graph.png")
    )
    plot_sensor_timeseries(
        U0.detach().cpu().numpy(), sensor_id=0,
        save_path=os.path.join(plot_dir, "sanity_timeseries.png")
    )
    plot_event_field(
        U0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_event_field.png")
    )
   
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

    plot_sensor_map(
        coords0.detach().cpu().numpy(),
        source_location=loc0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_sensor_map.png")
    )
    plot_sensor_graph(
        coords0.detach().cpu().numpy(),
        adj0[1].detach().cpu().numpy(),
        source_location=loc0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_sensor_graph.png")
    )
    plot_sensor_timeseries(
        U0.detach().cpu().numpy(), sensor_id=0,
        save_path=os.path.join(plot_dir, "sanity_timeseries.png")
    )
    plot_event_field(
        U0.detach().cpu().numpy(),
        save_path=os.path.join(plot_dir, "sanity_event_field.png")
    )

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
# ---------------------------------------------------------------------------
# Model, loss, optimizer configs
# ---------------------------------------------------------------------------
FAST_MODE = config.FAST_MODE
wno_config = config.WNO_CONFIG

print(f"\nFAST_MODE={FAST_MODE}  wno_config={wno_config}")

model = LocalizationModel(
    in_features=N_FEATURES,
    signal_length=T,
    use_real_data=config.USE_REAL_DATA,
    **wno_config,
).to(device)

loss_fn = Localizationloss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS)

N_EPOCHS = config.N_EPOCHS
display_epoch = config.DISPLAY_EPOCH
lambda_p = config.LAMBDA_P
lambda_c = config.LAMBDA_C

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    lambda_p=lambda_p,
    lambda_c=lambda_c,
    use_real_data=use_real_data,
)

training_history, best_state = trainer.train(epochs=N_EPOCHS, display_epoch=config.DISPLAY_EPOCH)

# Restore the best-validation-localization checkpoint before evaluation,
# since the final epoch is not guaranteed to be the best one for E(x).
if best_state is not None:
    model.load_state_dict(best_state)
    print("Loaded best checkpoint from memory.")
else:
    # Fallback: load from disk if memory state was lost
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'best_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best checkpoint from disk "
              f"(epoch {ckpt['epoch']}, val_loc={ckpt['val_loc']:.4f})")

logger = EvaluationLogger(output_dir=config.OUTPUT_DIR)
logger.log_config_snapshot(config)
logger.save_training_history(training_history)

evaluator = Evaluator(
    model=model,
    device=device,
    get_nearest_sensor_fn=get_nearest_sensor,
    print_results=False,
    use_real_data=config.USE_REAL_DATA,
    logger=logger,
)

print(f"\nProcessing evaluations... Exporting to directory: '{config.OUTPUT_DIR}'")

if use_real_data:
    fold_city_map = {
        "US_TO_EUROPE":          {"val": "Paris_Munich_Validation",
                                  "test": "Berlin_Rome_Unseen_Test"},
        "EUROPE_TO_US":          {"val": "Arizona_Nevada_Validation",
                                  "test": "California_NM_Utah_Unseen_Test"},
        "WITHIN_EU":             {"val": "Paris_Validation",
                                  "test": "Berlin_Unseen_Test"},
        "WITHIN_US_TEMPORAL":    {"val": "Arizona_2023_Validation",
                                  "test": "Arizona_2024_Unseen_Test"},
        "WITHIN_US_SPATIAL_S2M": {"val": "Arizona_2022_Validation",
                                  "test": "Arizona_2023_2024_Unseen_Test"},
        "WITHIN_US_SPATIAL_M2L": {"val": "Nevada_NM_Utah_Validation",
                                  "test": "California_Unseen_Test"},
        "WITHIN_US_SPATIAL_L2M": {"val":  "Nevada_NM_Utah_Validation",
                                  "test": "Arizona_Unseen_Test"},
        "TEMPORAL_EU":           {"val": "EU_All_2023_Validation",
                                  "test": "EU_All_2024_Unseen_Test"},
    }

    if DATA_SELECTOR not in fold_city_map:
        raise ValueError(
            f"Unknown fold '{DATA_SELECTOR}'. "
            f"Choose from: {list(fold_city_map.keys())}"
        )

    current_labels = fold_city_map[DATA_SELECTOR]
    
    # Execute the evaluations with 100% dynamic, accurate labels!
    out1 = evaluator.evaluate(val_loader, event_idx=0, label=current_labels["val"])
    out2 = evaluator.evaluate(test_loader, event_idx=0, label=current_labels["test"])
    
    from evaluate import run_causal_curves
    del_curve, ins_curve = run_causal_curves(model, val_loader, device=device, is_real_data=use_real_data, steps=4)  
    
    plt.figure(figsize=(6, 4))
    plt.plot(del_curve, color='red', marker='o', label='Deletion (Necessity) ↑')
    plt.plot(ins_curve, color='green', marker='s', linestyle='--', label='Insertion (Sufficiency) ↓')

    # 3. Add labels and display instantly
    plt.xlabel('Perturbed Sensor Ranks (Steps)'); plt.ylabel('Tracking Distance Error'); plt.legend()
    plt.title('Causal Faithfulness Explanatory Curves'); plt.grid(True, linestyle=':'); plt.show()
    
    # --------------------------------------------------------------------------- #
    # Real-World Self-Naming Publication Figure Generator
    # --------------------------------------------------------------------------- #
    print(f"\nGenerating publication figures for active fold validation: {current_labels['val']}")

    # Grab Event 0 from your active validation dataset object cleanly!
    # (Bypasses the DataLoader so dimensions are perfectly unbatched)
    U0, coords0, lap0, adj0, loc0, t0 = val_set[0]

    # Extract a single item out of your loader stream to calculate the model's true prediction coordinates
    val_batch_sample = next(iter(val_loader))
    U_b, coords_b, lap_b, adj_b, loc_b, _ = val_batch_sample

    model.eval()
    with torch.no_grad():
        pred_coords_e, _, _, _, _, _, _, _, _ = model(
            U_b.to(device), coords_b.to(device), lap_b.to(device), adj_b.to(device)
        )
    # Strip the extra leading batch axis out of your continuous prediction tensor
    loc_pred0 = pred_coords_e.squeeze(0).cpu().numpy()

    # TRIGGER THE UPDATED VISUALIZATION SUITE
    # Map out the true city station positions alongside the model's precise latitude/longitude guess!
    plot_sensor_map(
        coords=coords0.cpu().numpy(),
        source_location=loc0.cpu().numpy(),
        predicted_location=loc_pred0,
        title_label=current_labels["val"],
        save_path=os.path.join(plot_dir, "sensor_map.png")
    )

    # Plot the specific unaligned topology network graph for Feature Channel 0 (PM2.5)
    plot_sensor_graph(
        coords=coords0.cpu().numpy(),
        adjacency=adj0.cpu().numpy()[0], # Index 0 extracts your PM2.5 channel graph layout matrix
        feature_idx=0,
        source_location=loc0.cpu().numpy(),
        predicted_location=loc_pred0,
        title_label=current_labels["val"].split("_")[0], # Dynamically passes just the city name string "Paris"
        save_path=os.path.join(plot_dir, "sensor_graph.png")
    )

    # Track the 100-hour Ozone (O3 - Feature 5) history trend for station node 0
    plot_sensor_timeseries(
        U=U0.cpu().numpy(),
        sensor_id=0,
        feature_idx=5,
        save_path=os.path.join(plot_dir, "sensor_timeseries.png")
    )

    # Generate the full 2D spatiotemporal propagation wave profile for Temperature (Feature 8)
    plot_event_field(
        E=U0.cpu().numpy(),
        feature_idx=8,
        save_path=os.path.join(plot_dir, "sensor_field.png")
    )
    
else:
    # Keeps your original synthetic laboratory tags completely active for simulation tests
    out1 = evaluator.evaluate(val_loader, event_idx=0, label="Validation_Normal", is_real_data=False)
    out2 = evaluator.evaluate(adv_loader, event_idx=0, label="Validation_Adversarial_Decoy_1.5x", is_real_data=False)

    from evaluate import run_causal_curves
    del_curve_normal, ins_curve_normal = run_causal_curves(model, val_loader, device=device, is_real_data=False, steps=4)
    del_curve_adv, ins_curve_adv = run_causal_curves(model, adv_loader, device=device, is_real_data=False, steps=4)
    
    plt.figure(figsize=(6, 4))
    plt.plot(del_curve_normal, color='red', marker='o', label='Deletion (Normal)')
    plt.plot(ins_curve_normal, color='green', marker='s', linestyle='--', label='Insertion (Normal)')
    plt.plot(del_curve_adv, color='darkred', marker='^', label='Deletion (Adversarial)')
    plt.plot(ins_curve_adv, color='darkgreen', marker='v', linestyle=':', label='Insertion (Adversarial)')
    plt.xlabel('Perturbed Sensor Ranks (Steps)'); plt.ylabel('Localization Error')
    plt.legend(); plt.title('Causal Faithfulness Curves — Synthetic Data')
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(plot_dir, "causal_curves_synthetic.png"), dpi=100, bbox_inches="tight")
    plt.show()

# Import the enhanced analysis
from explainability_analysis import run_full_explainability_analysis

# For validation data (synthetic or real)
explainability_dir = os.path.join(config.OUTPUT_DIR, "explainability_plots")

if use_real_data:
    # For real data
    run_full_explainability_analysis(
        model=model,
        loader=val_loader,
        device=device,
        save_dir=explainability_dir,
        is_real_data=True,
        dataset_name=current_labels["val"],
        steps=4
    )
    
    run_full_explainability_analysis(
        model=model,
        loader=test_loader,
        device=device,
        save_dir=os.path.join(config.OUTPUT_DIR, "explainability_plots_test"),
        is_real_data=True,
        dataset_name=current_labels["test"],
        steps=4
    )
else:
    # For synthetic data
    run_full_explainability_analysis(
        model=model,
        loader=val_loader,
        device=device,
        save_dir=explainability_dir,
        is_real_data=False,
        dataset_name="Validation_Normal",
        steps=4
    )
    
    # Also for adversarial data
    if not use_real_data:
        run_full_explainability_analysis(
            model=model,
            loader=adv_loader,
            device=device,
            save_dir=os.path.join(config.OUTPUT_DIR, "explainability_plots_adversarial"),
            is_real_data=False,
            dataset_name="Validation_Adversarial",
            steps=4
        )
# =========================================================================== #
# RUN ENCAPSULATED IDW GEODESIC BASELINE
# =========================================================================== #
print("\n" + "="*80 + "\nRUNNING BASELINE 1: INVERSE DISTANCE WEIGHTING (IDW)\n" + "="*80)

# 1. Import the wrapper evaluator function directly
from baseline_idw import run_idw_evaluation

# 2. Trigger IDW with a single-line call matching your active dataset fold!
if use_real_data:
    run_idw_evaluation(val_loader, label=f"IDW_{current_labels['val']}", is_real_data=True)
    run_idw_evaluation(test_loader, label=f"IDW_{current_labels['test']}", is_real_data=True)
else:
    run_idw_evaluation(val_loader, label="IDW_Validation_Normal", is_real_data=False)
    
# =========================================================================== #
# RUN AND TRAIN BASELINE 2: SPATIOTEMPORAL BI-LSTM (STANDALONE PATH)
# =========================================================================== #
import torch.nn.functional as F

print("\n" + "="*80 + "\nTRAINING & EVALUATING BASELINE 2: SPATIOTEMPORAL BI-LSTM\n" + "="*80)

from baseline_lstm import LSTMIntensityBaseline
from baseline_trainer import BaselineTrainer  # <-- FIXED: Imports your clean standalone trainer

# Instantiate the LSTM Baseline architecture
lstm_model = LSTMIntensityBaseline(in_features=N_FEATURES, hidden_dim=32).to(device)

# Configure optimization parameters
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
lstm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lstm_optimizer, T_max=config.N_EPOCHS)

# Trigger the standalone BaselineTrainer
lstm_trainer = BaselineTrainer(
    model=lstm_model, loss_fn=F.mse_loss, optimizer=lstm_optimizer, scheduler=lstm_scheduler,
    train_loader=train_loader, val_loader=val_loader, device=device
)

_, lstm_best_state = lstm_trainer.train(epochs=config.N_EPOCHS, display_epoch=config.DISPLAY_EPOCH)
if lstm_best_state is not None:
    lstm_model.load_state_dict(lstm_best_state)
    print("Loaded best-val-loss checkpoint for BiLSTM.")

from baseline_trainer import run_baseline_evaluation
if use_real_data:
    run_baseline_evaluation(lstm_model, val_loader, label=f"BiLSTM_{current_labels['val']}", device=device, is_real_data=True, logger=logger)
    run_baseline_evaluation(lstm_model, test_loader, label=f"BiLSTM_{current_labels['test']}", device=device, is_real_data=True, logger=logger)
else:
    run_baseline_evaluation(lstm_model, val_loader, label="BiLSTM_Validation_Normal", device=device, is_real_data=False, logger=logger)
# =========================================================================== #
# RUN AND TRAIN BASELINE 3: SPATIOTEMPORAL FNO
# =========================================================================== #
print("\n" + "="*80 + "\nTRAINING & EVALUATING BASELINE 3: SPATIOTEMPORAL FNO\n" + "="*80)

from baseline_fno import FNOLocalizationModel
from baseline_trainer import BaselineTrainer, run_baseline_evaluation

# 1. Instantiate the FNO Baseline architecture
fno_model = FNOLocalizationModel(in_features=N_FEATURES, hidden_dim=32, modes=16).to(device)

# 2. Configure optimization parameters
fno_optimizer = torch.optim.Adam(fno_model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
fno_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fno_optimizer, T_max=config.N_EPOCHS)

# 3. Trigger the standalone BaselineTrainer
fno_trainer = BaselineTrainer(
    model=fno_model, loss_fn=F.mse_loss, optimizer=fno_optimizer, scheduler=fno_scheduler,
    train_loader=train_loader, val_loader=val_loader, device=device
)

# Run training epochs smoothly
_, fno_best_state = fno_trainer.train(epochs=config.N_EPOCHS, display_epoch=config.DISPLAY_EPOCH)
if fno_best_state is not None:
    fno_model.load_state_dict(fno_best_state)
    print("Loaded best-val-loss checkpoint for FNO.")

# 4. Evaluate using the independent Haversine evaluation function
if use_real_data:
    run_baseline_evaluation(fno_model, val_loader, label=f"FNO_{current_labels['val']}", device=device, is_real_data=True, logger=logger)
    run_baseline_evaluation(fno_model, test_loader, label=f"FNO_{current_labels['test']}", device=device, is_real_data=True, logger=logger)
else:
    run_baseline_evaluation(fno_model, val_loader, label="FNO_Validation_Normal", device=device, is_real_data=False, logger=logger)

# =========================================================================== #
# RUN AND TRAIN BASELINE 4: SPATIOTEMPORAL TRANSFORMER ENCODER
# =========================================================================== #
print("\n" + "="*80 + "\nTRAINING & EVALUATING BASELINE 4: SPATIOTEMPORAL TRANSFORMER\n" + "="*80)

import torch.nn.functional as F
from baseline_transformer import TransformerLocalizationModel
from baseline_trainer import BaselineTrainer, run_baseline_evaluation

# 1. Instantiate the Transformer baseline (using 4 attention heads)
trans_model = TransformerLocalizationModel(in_features=N_FEATURES, hidden_dim=64, n_heads=4, signal_length=T).to(device)

# 2. Configure optimization parameters
trans_optimizer = torch.optim.Adam(trans_model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
trans_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trans_optimizer, T_max=config.N_EPOCHS)

# 3. Trigger the standalone BaselineTrainer using functional MSE loss
trans_trainer = BaselineTrainer(
    model=trans_model, 
    loss_fn=F.mse_loss, 
    optimizer=trans_optimizer, 
    scheduler=trans_scheduler,
    train_loader=train_loader, 
    val_loader=val_loader, 
    device=device
)

# Run training epochs smoothly
_, trans_best_state = trans_trainer.train(epochs=config.N_EPOCHS, display_epoch=config.DISPLAY_EPOCH)
if trans_best_state is not None:
    trans_model.load_state_dict(trans_best_state)
    print("Loaded best-val-loss checkpoint for Transformer.")

# 4. Evaluate using the independent Haversine evaluation function
if use_real_data:
    run_baseline_evaluation(trans_model, val_loader, label=f"Transformer_{current_labels['val']}", device=device, is_real_data=True, logger=logger)
    run_baseline_evaluation(trans_model, test_loader, label=f"Transformer_{current_labels['test']}", device=device, is_real_data=True, logger=logger)
else:
    run_baseline_evaluation(trans_model, val_loader, label="Transformer_Validation_Normal", device=device, is_real_data=False, logger=logger)

# =========================================================================== #
# RUN ADVERSARIAL SUPPRESSION TESTS ON ALL BENCHMARKS
# =========================================================================== #
if not use_real_data:
    print("\n" + "="*80 + "\nRUNNING ADVERSARIAL STRESS TESTING (DECOY SUPPRESSION CHANNELS)\n" + "="*80)
    from baseline_trainer import run_baseline_evaluation
    
    run_idw_evaluation(adv_loader, label="IDW_Adversarial_Decoy_1.5x", is_real_data=False, logger=logger)
    
    # Pass the adversarial decoy loader stream through your benchmark deck cleanly
    run_baseline_evaluation(lstm_model, adv_loader, label="BiLSTM_Adversarial_Decoy_1.5x", device=device, is_real_data=False, logger=logger)
    run_baseline_evaluation(fno_model, adv_loader, label="FNO_Adversarial_Decoy_1.5x", device=device, is_real_data=False, logger=logger)
    run_baseline_evaluation(trans_model, adv_loader, label="Transformer_Adversarial_Decoy_1.5x", device=device, is_real_data=False, logger=logger)
    
    print("\nAdversarial benchmarking blocks successfully appended to disk logs!")