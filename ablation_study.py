# -*- coding: utf-8 -*-
"""
ablation_study.py -- Standalone ablation runner for WNO Localization Framework.
Ablations available:
  1. P(x) component removal  -- zero out one of frequency/temporal/neighbor
  2. Graph layer ablation    -- bypass PerFeatureGraphSpectralLayer
Usage:
  python ablation_study.py --checkpoint_dir logs/2026-08-01_120000 --ablation all
Author: generated for WNO paper ablation studies
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

import config
from model import LocalizationModel
from evaluate import Evaluator

_on_kaggle = os.path.exists('/kaggle/working')
try:
    import google.colab
    _on_colab = True
except ImportError:
    _on_colab = False

# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='WNO Ablation Study Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[40, 41, 42, 43, 44])
    parser.add_argument('--checkpoint_name', type=str,
                        default='best_checkpoint.pt')
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['px_components', 'graph_layer', 'all'])
    parser.add_argument('--real_data',
                        type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--fold', type=str, default='WITHIN_EU',
                        choices=['US_TO_EUROPE', 'WITHIN_EU',
                                 'WITHIN_US_TEMPORAL',
                                 'WITHIN_US_SPATIAL_S2M',
                                 'WITHIN_US_SPATIAL_M2L',
                                 'WITHIN_US_SPATIAL_L2M',
                                 'TEMPORAL_EU', 'EUROPE_TO_US'])
    parser.add_argument('--datafolder', type=str, default=None)
    parser.add_argument('--n_events', type=int, default=500)
    parser.add_argument('--output_dir', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args

# ============================================================================
# CHECKPOINT DISCOVERY
# ============================================================================
def find_checkpoint_dir():
    if _on_kaggle:
        log_base = '/kaggle/working/logs'
    elif _on_colab:
        log_base = '/content/drive/MyDrive/WNO Project/version 5/logs'
    else:
        log_base = 'logs'
    if not os.path.exists(log_base):
        raise FileNotFoundError(f"No logs directory at '{log_base}'.")
    runs = sorted([
        d for d in os.listdir(log_base)
        if os.path.isdir(os.path.join(log_base, d))
    ])
    if not runs:
        raise FileNotFoundError(f"No run directories in '{log_base}'.")
    latest = os.path.join(log_base, runs[-1])
    print(f"[AutoDiscover] Using: {latest}")
    return latest

def load_checkpoint(checkpoint_dir, seed, checkpoint_name, device):
    seed_dir  = os.path.join(checkpoint_dir, f'seed{seed}')
    ckpt_path = os.path.join(seed_dir, checkpoint_name)
    cfg_path  = os.path.join(seed_dir, 'config_snapshot.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Available: {os.listdir(seed_dir) if os.path.exists(seed_dir) else 'dir missing'}"
        )

    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"  [Seed {seed}] fold={cfg.get('DATA_SELECTOR')} "
              f"lambda_p={cfg.get('LAMBDA_P')}")
    else:
        print(f"  [Seed {seed}] WARNING: no config_snapshot.json")

    wno_cfg    = cfg.get('WNO_CONFIG', {
        'wno_width': 32, 'wno_levels': 3,
        'wno_layers': 2, 'wavelet': 'db4'
    })
    n_features = cfg.get('N_FEATURES', config.N_FEATURES)
    T          = cfg.get('T', config.T)
    use_real   = cfg.get('USE_REAL_DATA', config.USE_REAL_DATA)

    model = LocalizationModel(
        in_features=n_features,
        use_real_data=use_real,
        wno_width=wno_cfg.get('wno_width', 32),
        wno_levels=wno_cfg.get('wno_levels', 3),
        wno_layers=wno_cfg.get('wno_layers', 2),
        signal_length=T,
        wavelet=wno_cfg.get('wavelet', 'db4'),
    ).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"  [Seed {seed}] epoch={ckpt.get('epoch','?')} "
          f"val_loc={ckpt.get('val_loc', float('nan')):.4f}")
    return model, cfg

# ============================================================================
# DATA LOADERS
# ============================================================================
def get_synthetic_loader(n_events, seed):
    from dataset import EventDataset
    from synthetic_data import SyntheticEventGenerator
    from torch.utils.data import DataLoader, random_split

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = SyntheticEventGenerator(
        n_sensors=config.N_SENSORS,
        n_timesteps=config.T,
        n_features=config.N_FEATURES,
        random_seed=seed,
    )
    coords = generator.generate_sensor_locations()

    normal_dataset = EventDataset(
        generator, num_events=n_events, coords=coords, lazy=False,
        k_neighbors=4, graph_sigma=0.2, coverage_prob=0.6,
        mask_seed=seed,
        velocity=[0.1, 0.02, 0.05],
        broadening=[0.5, 4.0, 2.0],
    )
    n_val   = int(0.2 * len(normal_dataset))
    n_train = len(normal_dataset) - n_val
    _, val_set = random_split(
        normal_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(val_set, batch_size=8, shuffle=False)

def get_real_loader(fold, datafolder, seed):
    from real_dataset_loader import (scan_and_process_folder,
                                     build_fold, regions_to_dataset)
    from torch.utils.data import DataLoader

    data_path   = datafolder or config.DATAFOLDER
    all_regions = scan_and_process_folder(data_path, force_reprocess=False)
    _, val_regions, _, labels = build_fold(all_regions, fold)
    val_dataset = regions_to_dataset(val_regions, use_real_data=True)
    return DataLoader(val_dataset, batch_size=1, shuffle=False), labels

# ============================================================================
# CORE EVALUATION
# E(x), E(t), P(x) — all computed in one pass
# ============================================================================
def run_one_eval(model, loader, device, use_real_data,
                 ablate_component=None):
    """
    ablate_component: None | 'frequency' | 'temporal' | 'neighbor'

    Returns dict with:
      Ex_mean, Ex_std       — spatial localization error
      Et_mean, Et_std       — temporal onset error (hours)
      Top1, Top3, Top5      — P(x) top-k accuracy (%)
      Proximity             — P(x) proximity accuracy (%)
    """
    all_ex   = []
    all_et   = []
    all_top1 = []
    all_top3 = []
    all_top5 = []
    all_prox = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            U, coords_batch, lap_batch, adj_batch, location, time_batch = batch
            U            = U.to(device)
            coords_batch = coords_batch.to(device)
            lap_batch    = lap_batch.to(device)
            adj_batch    = adj_batch.to(device)
            location     = location.to(device)
            time_batch   = time_batch.to(device)

            # ── Forward pass ─────────────────────────────────────────────────
            (coords_pred_e, time_pred, intensity, Px,
             frequency_score, temporal_score, neighbor_score,
             band_energy, alpha) = model(
                U, coords_batch, lap_batch, adj_batch
            )

            # ── P(x) component ablation ──────────────────────────────────────
            if ablate_component is not None:
                if ablate_component == 'frequency':
                    frequency_score = torch.ones_like(frequency_score) * 0.5
                elif ablate_component == 'temporal':
                    temporal_score  = torch.ones_like(temporal_score)  * 0.5
                elif ablate_component == 'neighbor':
                    neighbor_score  = torch.ones_like(neighbor_score)  * 0.5

                px_features = torch.stack([
                    frequency_score.contiguous(),
                    temporal_score.contiguous(),
                    neighbor_score.contiguous(),
                ], dim=-1)
                s_temp = torch.clamp(
                    torch.exp(model.log_temp_spatial),
                    min=0.02, max=0.10
                )
                Px = model.px_head(px_features).squeeze(-1)
                Px = torch.softmax(Px / s_temp, dim=1)

            B, N, _ = coords_batch.shape

            # ── E(x) ─────────────────────────────────────────────────────────
            if use_real_data:
                ex_err = Evaluator._compute_haversine_distance(
                    coords_pred_e, location
                ).cpu().numpy()
            else:
                ex_err = torch.norm(
                    coords_pred_e - location, dim=1
                ).cpu().numpy()
            all_ex.extend(ex_err.tolist())

            # ── E(t) ─────────────────────────────────────────────────────────
            et_err = torch.abs(
                time_pred.squeeze(-1) - time_batch.float().squeeze()
            ).cpu().numpy()
            et_err = np.atleast_1d(et_err.flatten())
            all_et.extend(et_err.tolist())

            # ── P(x) top-k ───────────────────────────────────────────────────
            pred_sensor = Px.argmax(dim=1)   # (B,)

            def nearest_k(k):
                k_eff = min(k, N)
                if use_real_data:
                    R    = 6371.0
                    lat1 = torch.deg2rad(location[:, 0]).unsqueeze(1)
                    lon1 = torch.deg2rad(location[:, 1]).unsqueeze(1)
                    lat2 = torch.deg2rad(coords_batch[:, :, 0])
                    lon2 = torch.deg2rad(coords_batch[:, :, 1])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a    = (torch.sin(dlat / 2) ** 2
                            + torch.cos(lat1) * torch.cos(lat2)
                            * torch.sin(dlon / 2) ** 2)
                    dist = 2 * R * torch.asin(torch.sqrt(a.clamp(0, 1)))
                else:
                    dist = torch.norm(
                        coords_batch - location.unsqueeze(1), dim=2
                    )
                k_nearest = dist.topk(k_eff, largest=False).indices
                correct   = (pred_sensor.unsqueeze(1) == k_nearest
                             ).any(dim=1).float()
                return correct

            all_top1.extend(nearest_k(1).cpu().tolist())
            all_top3.extend(nearest_k(3).cpu().tolist())
            all_top5.extend(nearest_k(5).cpu().tolist())

            # ── Proximity ────────────────────────────────────────────────────
            batch_idx = torch.arange(B, device=device)
            if use_real_data:
                R    = 6371.0
                lat1 = torch.deg2rad(location[:, 0]).unsqueeze(1)
                lon1 = torch.deg2rad(location[:, 1]).unsqueeze(1)
                lat2 = torch.deg2rad(coords_batch[:, :, 0])
                lon2 = torch.deg2rad(coords_batch[:, :, 1])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a    = (torch.sin(dlat / 2) ** 2
                        + torch.cos(lat1) * torch.cos(lat2)
                        * torch.sin(dlon / 2) ** 2)
                dist_to_loc = 2 * R * torch.asin(
                    torch.sqrt(a.clamp(0, 1))
                )
                threshold = 15.0
            else:
                dist_to_loc = torch.norm(
                    coords_batch - location.unsqueeze(1), dim=2
                )
                threshold = 0.15

            source_sensor = dist_to_loc.argmin(dim=1)
            true_coords   = coords_batch[batch_idx, source_sensor]
            pred_coords   = coords_batch[batch_idx, pred_sensor]

            if use_real_data:
                spatial_dist = Evaluator._compute_haversine_distance(
                    pred_coords, true_coords
                )
            else:
                spatial_dist = torch.norm(
                    pred_coords - true_coords, dim=1
                )

            prox = (spatial_dist <= threshold).float()
            all_prox.extend(prox.cpu().tolist())

    return {
        'Ex_mean':   float(np.mean(all_ex)),
        'Ex_std':    float(np.std(all_ex)),
        'Et_mean':   float(np.mean(all_et)),
        'Et_std':    float(np.std(all_et)),
        'Top1':      float(np.mean(all_top1) * 100),
        'Top3':      float(np.mean(all_top3) * 100),
        'Top5':      float(np.mean(all_top5) * 100),
        'Proximity': float(np.mean(all_prox) * 100),
    }

# ============================================================================
# HELPER — empty results dict
# ============================================================================
def _empty_results(keys):
    return {k: {m: [] for m in [
        'Ex_mean', 'Ex_std', 'Et_mean', 'Et_std',
        'Top1', 'Top3', 'Top5', 'Proximity'
    ]} for k in keys}

def _summarize(results):
    summary = {}
    for label, r in results.items():
        row = {}
        for k in r:
            vals = r[k]
            if vals:
                row[k] = {
                    'mean': float(np.mean(vals)),
                    'std':  float(np.std(vals)),
                }
        summary[label] = row
    return summary

def _print_summary(summary):
    print(f"\n{'Variant':25s} | {'E(x)':>12s} | {'E(t)':>12s} | "
          f"{'Top-1':>8s} | {'Top-5':>8s} | {'Proximity':>10s}")
    print("-"*85)
    for label, row in summary.items():
        ex = f"{row['Ex_mean']['mean']:.4f}±{row['Ex_mean']['std']:.4f}"
        et = f"{row['Et_mean']['mean']:.4f}±{row['Et_mean']['std']:.4f}"
        t1 = f"{row['Top1']['mean']:.1f}±{row['Top1']['std']:.1f}%"
        t5 = f"{row['Top5']['mean']:.1f}±{row['Top5']['std']:.1f}%"
        pr = f"{row['Proximity']['mean']:.1f}±{row['Proximity']['std']:.1f}%"
        print(f"{label:25s} | {ex:>12s} | {et:>12s} | "
              f"{t1:>8s} | {t5:>8s} | {pr:>10s}")

def _save_results(summary, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f'{filename}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, f'{filename}.csv')
    with open(csv_path, 'w') as f:
        f.write("Variant,Ex_mean,Ex_std,Et_mean,Et_std,"
                "Top1_mean,Top1_std,Top3_mean,Top3_std,"
                "Top5_mean,Top5_std,Proximity_mean,Proximity_std\n")
        for label, row in summary.items():
            f.write(
                f"{label},"
                f"{row['Ex_mean']['mean']:.4f},{row['Ex_mean']['std']:.4f},"
                f"{row['Et_mean']['mean']:.4f},{row['Et_mean']['std']:.4f},"
                f"{row['Top1']['mean']:.2f},{row['Top1']['std']:.2f},"
                f"{row['Top3']['mean']:.2f},{row['Top3']['std']:.2f},"
                f"{row['Top5']['mean']:.2f},{row['Top5']['std']:.2f},"
                f"{row['Proximity']['mean']:.2f},"
                f"{row['Proximity']['std']:.2f}\n"
            )
    print(f"\nSaved: {json_path}")
    print(f"CSV:   {csv_path}")

# ============================================================================
# ABLATION: P(x) COMPONENT REMOVAL
# ============================================================================
def ablation_px_components(checkpoint_dir, seeds, checkpoint_name,
                            use_real_data, fold, datafolder,
                            n_events, device, output_dir):
    print("\n" + "="*70)
    print("ABLATION: P(x) COMPONENT REMOVAL")
    print("="*70)

    components = [None, 'frequency', 'temporal', 'neighbor']
    labels_map = {
        None:        'Full P(x)',
        'frequency': 'w/o Frequency Score',
        'temporal':  'w/o Temporal Score',
        'neighbor':  'w/o Neighbor Score',
    }

    results = _empty_results(list(labels_map.values()))

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        try:
            model, cfg = load_checkpoint(
                checkpoint_dir, seed, checkpoint_name, device
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        loader = (get_real_loader(fold, datafolder, seed)[0]
                  if use_real_data
                  else get_synthetic_loader(n_events, seed))

        for comp in components:
            label   = labels_map[comp]
            metrics = run_one_eval(
                model, loader, device, use_real_data,
                ablate_component=comp
            )
            for k, v in metrics.items():
                results[label][k].append(v)
            print(f"  {label:25s} | "
                  f"E(x):{metrics['Ex_mean']:.4f} "
                  f"E(t):{metrics['Et_mean']:.4f} "
                  f"Top1:{metrics['Top1']:.1f}% "
                  f"Top5:{metrics['Top5']:.1f}% "
                  f"Prox:{metrics['Proximity']:.1f}%")

    print("\n" + "="*70)
    print("SUMMARY (mean ± std across seeds)")
    print("="*70)
    summary = _summarize(results)
    _print_summary(summary)
    _save_results(summary, output_dir, 'px_component_ablation')
    return summary

# ============================================================================
# ABLATION: GRAPH LAYER
# ============================================================================
def patch_model_for_graph_ablation(model):
    original_graph_layer = model.graph_layer

    class IdentityGraphLayer(torch.nn.Module):
        def forward(self, U, lap):
            return U

    model.graph_layer = IdentityGraphLayer()
    return model, original_graph_layer

def ablation_graph_layer(checkpoint_dir, seeds, checkpoint_name,
                         use_real_data, fold, datafolder,
                         n_events, device, output_dir):
    print("\n" + "="*70)
    print("ABLATION: GRAPH SPECTRAL LAYER")
    print("="*70)

    results = _empty_results(['Full Model', 'No Graph Layer'])

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        try:
            model, cfg = load_checkpoint(
                checkpoint_dir, seed, checkpoint_name, device
            )
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        loader = (get_real_loader(fold, datafolder, seed)[0]
                  if use_real_data
                  else get_synthetic_loader(n_events, seed))

        # Full model
        m_full = run_one_eval(model, loader, device, use_real_data)
        for k, v in m_full.items():
            results['Full Model'][k].append(v)
        print(f"  {'Full Model':20s} | "
              f"E(x):{m_full['Ex_mean']:.4f} "
              f"E(t):{m_full['Et_mean']:.4f} "
              f"Prox:{m_full['Proximity']:.1f}%")

        # No graph layer
        model, original_layer = patch_model_for_graph_ablation(model)
        m_no_graph = run_one_eval(model, loader, device, use_real_data)
        model.graph_layer = original_layer  # restore
        for k, v in m_no_graph.items():
            results['No Graph Layer'][k].append(v)
        print(f"  {'No Graph Layer':20s} | "
              f"E(x):{m_no_graph['Ex_mean']:.4f} "
              f"E(t):{m_no_graph['Et_mean']:.4f} "
              f"Prox:{m_no_graph['Proximity']:.1f}%")

    print("\n" + "="*70)
    print("SUMMARY (mean ± std across seeds)")
    print("="*70)
    summary = _summarize(results)
    _print_summary(summary)
    _save_results(summary, output_dir, 'graph_layer_ablation')
    return summary

# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()
    device = config.get_device()

    if args.checkpoint_dir is None:
        checkpoint_dir = find_checkpoint_dir()
    else:
        checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Not found: {checkpoint_dir}"
            )

    output_dir = args.output_dir or os.path.join(
        checkpoint_dir, 'ablation'
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nCheckpoint: {checkpoint_dir}")
    print(f"Seeds:      {args.seeds}")
    print(f"Ablation:   {args.ablation}")
    print(f"Real data:  {args.real_data}")
    if args.real_data:
        print(f"Fold:       {args.fold}")
    print(f"Output:     {output_dir}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    kwargs = dict(
        checkpoint_dir=checkpoint_dir,
        seeds=args.seeds,
        checkpoint_name=args.checkpoint_name,
        use_real_data=args.real_data,
        fold=args.fold,
        datafolder=args.datafolder,
        n_events=args.n_events,
        device=device,
        output_dir=output_dir,
    )

    if args.ablation in ('px_components', 'all'):
        ablation_px_components(**kwargs)

    if args.ablation in ('graph_layer', 'all'):
        ablation_graph_layer(**kwargs)

    print(f"\nDone. Results in: {output_dir}")

if __name__ == '__main__':
    main()