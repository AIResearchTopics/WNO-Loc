# -*- coding: utf-8 -*-
"""
eval_only.py — Reload saved checkpoints and rerun evaluation only.
Writes to evaluation_metrics_corrected.csv — never touches original.
Usage:
    python eval_only.py --fold TEMPORAL_EU \
        --logs_path "Final Result/TEMPORAL_EU/2026-07-31_093244"
"""
import os
import csv
import json
import glob
import re
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from model import LocalizationModel
from evaluate import Evaluator
from evaluationlogger import EvaluationLogger
from real_dataset_loader import (
    scan_and_process_folder, build_fold, regions_to_dataset
)

# ============================================================================
# CORRECTED LOGGER — writes to evaluation_metrics_corrected.csv only
# ============================================================================
class CorrectedLogger(EvaluationLogger):
    def __init__(self, output_dir, overwrite=True):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.txt_path = os.path.join(
            self.output_dir, "evaluation_report_corrected.txt"
        )
        self.csv_path = os.path.join(
            self.output_dir, "evaluation_metrics_corrected.csv"
        )

        # Always overwrite — never append
        with open(self.csv_path, mode='w',
                  newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Label",
                "Ex_Loc_Mean", "Ex_Loc_Std",
                "Ex_Loc_Median", "Ex_Loc_Normalized",
                "Px_Loc_Mean", "Px_Loc_Std",
                "Px_Loc_Median", "Px_Loc_Normalized",
                "Peak_Loc_Mean", "Peak_Loc_Std", "Peak_Loc_Median",
                "Ex_Time_Mean", "Ex_Time_Std", "Ex_Time_P90",
                "Px_Top1_Acc", "Px_Top3_Acc", "Px_Top5_Acc",
                "Px_Top10_Acc", "Px_Proximity_Acc",
                "Explainability_Fidelity_Delta",
                "Sensor_Dist_Mean", "Sensor_Dist_Std",
                "Sensor_Idx_Error",
                "Freq_Mean", "Freq_Std", "Freq_Max",
                "Temp_Mean", "Temp_Std", "Temp_Max",
                "Neigh_Mean", "Neigh_Std", "Neigh_Max",
                "Target_Event_Idx",
                "Event_N_True_Sensor",
                "Event_N_Pred_Sensor",
                "Event_N_Wavelet_Weights",
            ])

# ============================================================================
# ARGS
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, required=True,
                        choices=[
                            'US_TO_EUROPE', 'WITHIN_EU',
                            'WITHIN_US_TEMPORAL',
                            'WITHIN_US_SPATIAL_S2M',
                            'WITHIN_US_SPATIAL_M2L',
                            'WITHIN_US_SPATIAL_L2M',
                            'TEMPORAL_EU', 'EUROPE_TO_US'
                        ])
    parser.add_argument('--logs_path', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[40, 41, 42, 43, 44])
    parser.add_argument('--checkpoint_name', type=str,
                        default='best_checkpoint.pt')
    parser.add_argument('--datafolder', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args

# ============================================================================
# FOLD LABELS
# ============================================================================
FOLD_LABEL_MAP = {
    "US_TO_EUROPE": {
        "val":  "Paris_Munich_Validation",
        "test": "Berlin_Rome_Unseen_Test"
    },
    "EUROPE_TO_US": {
        "val":  "Arizona_Nevada_Validation",
        "test": "California_NM_Utah_Unseen_Test"
    },
    "WITHIN_EU": {
        "val":  "Paris_Validation",
        "test": "Berlin_Unseen_Test"
    },
    "WITHIN_US_TEMPORAL": {
        "val":  "Arizona_2023_Validation",
        "test": "Arizona_2024_Unseen_Test"
    },
    "WITHIN_US_SPATIAL_S2M": {
        "val":  "Arizona_2022_Validation",
        "test": "Arizona_2023_2024_Unseen_Test"
    },
    "WITHIN_US_SPATIAL_M2L": {
        "val":  "Nevada_NM_Utah_Validation",
        "test": "California_Unseen_Test"
    },
    "WITHIN_US_SPATIAL_L2M": {
        "val":  "Nevada_NM_Utah_Validation",
        "test": "Arizona_Unseen_Test"
    },
    "TEMPORAL_EU": {
        "val":  "EU_All_2023_Validation",
        "test": "EU_All_2024_Unseen_Test"
    },
}

# ============================================================================
# NEAREST SENSOR
# ============================================================================
def get_nearest_sensor(coords_batch, true_location,
                       is_real_data=True):
    if is_real_data:
        R = 6371.0
        lat1 = torch.deg2rad(true_location[:, 0])
        lon1 = torch.deg2rad(true_location[:, 1])
        lat2 = torch.deg2rad(coords_batch[:, :, 0])
        lon2 = torch.deg2rad(coords_batch[:, :, 1])
        dlat = lat2 - lat1.unsqueeze(1)
        dlon = lon2 - lon1.unsqueeze(1)
        a = (torch.sin(dlat/2)**2
             + torch.cos(lat1.unsqueeze(1))
             * torch.cos(lat2)
             * torch.sin(dlon/2)**2)
        dist = 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))
    else:
        dist = torch.norm(
            coords_batch - true_location.unsqueeze(1), dim=2
        )
    return dist.argmin(dim=1)

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================
def load_checkpoint(logs_path, seed, checkpoint_name, device):
    seed_dir  = os.path.join(logs_path, f'seed{seed}')
    ckpt_path = os.path.join(seed_dir, checkpoint_name)
    cfg_path  = os.path.join(seed_dir, 'config_snapshot.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Not found: {ckpt_path}")

    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    wno_cfg = cfg.get('WNO_CONFIG', {
        'wno_width': 32, 'wno_levels': 3,
        'wno_layers': 2, 'wavelet': 'db4'
    })
    T = cfg.get('T', config.T)

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    # Infer N_FEATURES from checkpoint weights
    if 'graph_layer.order_weights' in state:
        n_features = state['graph_layer.order_weights'].shape[0]
    else:
        n_features = cfg.get('N_FEATURES', config.N_FEATURES)

    print(f"  [Seed {seed}] N_FEATURES={n_features} T={T} "
          f"fold={cfg.get('DATA_SELECTOR')} "
          f"lambda_p={cfg.get('LAMBDA_P')}")

    model = LocalizationModel(
        in_features=n_features,
        use_real_data=True,
        wno_width=wno_cfg.get('wno_width', 32),
        wno_levels=wno_cfg.get('wno_levels', 3),
        wno_layers=wno_cfg.get('wno_layers', 2),
        signal_length=T,
        wavelet=wno_cfg.get('wavelet', 'db4'),
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f"  [Seed {seed}] epoch={ckpt.get('epoch','?')} "
          f"val_loc={ckpt.get('val_loc', float('nan')):.4f}")
    return model, seed_dir

# ============================================================================
# AGGREGATE CORRECTED CSVS
# ============================================================================
def aggregate_corrected(logs_path, fold, labels):
    """
    Read all evaluation_metrics_corrected.csv files across seeds
    and save a summary to the run-level folder.
    """
    files = glob.glob(
        os.path.join(logs_path, 'seed*',
                     'evaluation_metrics_corrected.csv')
    )
    if not files:
        print("No corrected CSV files found to aggregate.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, engine='python')
            seed_match = re.search(r'seed(\d+)', f)
            if seed_match:
                df['seed'] = int(seed_match.group(1))
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")

    if not dfs:
        print("No data to aggregate.")
        return

    all_data = pd.concat(dfs, ignore_index=True)

    # Metrics to summarize
    metrics = [
        'Ex_Loc_Mean', 'Ex_Time_Mean', 'Ex_Time_P90',
        'Px_Top1_Acc', 'Px_Top3_Acc', 'Px_Top5_Acc',
        'Px_Proximity_Acc', 'Explainability_Fidelity_Delta',
        'Peak_Loc_Mean', 'Px_Loc_Mean',
    ]

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS — {fold}")
    print(f"{'='*60}")

    summary_rows = []
    for split, label in labels.items():
        split_df = all_data[all_data['Label'] == label]
        if split_df.empty:
            print(f"  No data for label: {label}")
            continue

        print(f"\n  {split.upper()}: {label} "
              f"({len(split_df)} seeds)")
        print(f"  {'Metric':<35} {'Mean':>10} {'Std':>10}")
        print(f"  {'-'*55}")

        for metric in metrics:
            if metric not in split_df.columns:
                continue
            vals = pd.to_numeric(
                split_df[metric], errors='coerce'
            ).dropna()
            if vals.empty:
                continue
            print(f"  {metric:<35} "
                  f"{vals.mean():>10.4f} "
                  f"{vals.std():>10.4f}")
            summary_rows.append({
                'Fold':   fold,
                'Split':  split,
                'Label':  label,
                'Metric': metric,
                'Mean':   vals.mean(),
                'Std':    vals.std(),
                'N':      len(vals),
            })

    # Save summary
    summary_path = os.path.join(
        logs_path, 'corrected_summary.csv'
    )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            summary_path, index=False
        )
        print(f"\n  Summary saved -> {summary_path}")

    # Save full aggregated data
    full_path = os.path.join(
        logs_path, 'corrected_all_seeds.csv'
    )
    all_data.to_csv(full_path, index=False)
    print(f"  Full data saved -> {full_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    args   = parse_args()
    device = config.get_device()

    print(f"\n{'='*60}")
    print(f"EVAL-ONLY — {args.fold}")
    print(f"Logs: {args.logs_path}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}")

    # Load data once — shared across all seeds
    datafolder  = args.datafolder or config.DATAFOLDER
    all_regions = scan_and_process_folder(
        data_folder=datafolder,
        window_length=config.T,
        pre_event_hours=48,
        max_tier=3,
        exclude_diffuse=False,
        max_wildfire_dist_km=40.0,
        force_reprocess=False,
    )

    _, val_regions, test_regions, _ = build_fold(
        all_regions, fold=args.fold
    )

    labels = FOLD_LABEL_MAP[args.fold]

    # Run each seed
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        try:
            model, seed_dir = load_checkpoint(
                args.logs_path, seed,
                args.checkpoint_name, device
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # CorrectedLogger writes to evaluation_metrics_corrected.csv
        logger = CorrectedLogger(output_dir=seed_dir)

        evaluator = Evaluator(
            model=model,
            device=device,
            get_nearest_sensor_fn=get_nearest_sensor,
            print_results=True,
            use_real_data=True,
            logger=logger,
        )
        
        val_set  = regions_to_dataset(
            val_regions,  k_neighbors=4,
            graph_sigma=0.2, coverage_prob=0.8,
            mask_seed=seed
        )
        test_set = regions_to_dataset(
            test_regions, k_neighbors=4,
            graph_sigma=0.2, coverage_prob=0.8,
            mask_seed=seed
        )

        val_loader  = DataLoader(
            val_set,  batch_size=1, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False
        )

        print(f"\n  Val: {labels['val']}")
        evaluator.evaluate(
            val_loader,
            label=labels['val'],
            event_idx=0,
            is_real_data=True
        )

        print(f"\n  Test: {labels['test']}")
        evaluator.evaluate(
            test_loader,
            label=labels['test'],
            event_idx=0,
            is_real_data=True
        )

        print(f"\n  Seed {seed} done -> "
              f"{os.path.join(seed_dir, 'evaluation_metrics_corrected.csv')}")

    # Aggregate all seeds
    print(f"\n{'='*60}")
    print(f"AGGREGATING ALL SEEDS")
    print(f"{'='*60}")
    aggregate_corrected(args.logs_path, args.fold, labels)

    print(f"\n{'='*60}")
    print(f"COMPLETE. Files in: {args.logs_path}")
    print(f"  - seed*/evaluation_metrics_corrected.csv")
    print(f"  - corrected_summary.csv")
    print(f"  - corrected_all_seeds.csv")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()