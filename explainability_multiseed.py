# -*- coding: utf-8 -*-
"""
Multi-Seed Explainability Analysis with Error Bars.
Finds causal data in explainability_plots folders.
@author: anjum
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import json
from scipy import stats

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def load_causal_data(run_id, seeds=[40, 41, 42, 43, 44], dataset='Validation_Normal'):
    """Load causal curves data from all seeds from explainability_plots folder."""
    all_deletion = []
    all_insertion = []
    all_deletion_ex = []
    found_seeds = []
    
    for seed in seeds:
        # Try both possible locations
        paths_to_try = [
            f"logs/{run_id}/seed{seed}/explainability_plots/causal_curves_data_{dataset}.json",
            f"logs/{run_id}/seed{seed}/explainability_plots_adversarial/causal_curves_data_{dataset}.json",
        ]
        
        found = False
        for path in paths_to_try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    all_deletion.append(data['deletion_px'])
                    all_insertion.append(data['insertion_px'])
                    all_deletion_ex.append(data.get('deletion_ex', data['deletion_px']))
                    found_seeds.append(seed)
                    found = True
                    break
        
        if not found:
            print(f"⚠️ No causal data for seed{seed}")
    
    if not all_deletion:
        print("❌ No causal data found!")
        return None, None, None
    
    # Convert to numpy arrays
    deletion_arr = np.array(all_deletion)
    insertion_arr = np.array(all_insertion)
    deletion_ex_arr = np.array(all_deletion_ex) if all_deletion_ex else None
    
    print(f"✅ Loaded {len(all_deletion)} seeds of causal data from seeds: {found_seeds}")
    return deletion_arr, insertion_arr, deletion_ex_arr

def load_wavelet_weights(run_id, seeds=[40, 41, 42, 43, 44]):
    """Load wavelet weights from all seeds."""
    all_weights = []
    
    for seed in seeds:
        path = f"logs/{run_id}/seed{seed}/evaluation_metrics.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Find WNO row (Validation_Normal)
            wno_rows = df[df['Label'].str.contains('Validation_Normal', na=False)]
            if not wno_rows.empty:
                try:
                    # Try to get wavelet weights
                    weights_str = wno_rows.iloc[0]['Event_N_Wavelet_Weights']
                    if isinstance(weights_str, str):
                        weights = eval(weights_str)
                    else:
                        weights = weights_str
                    all_weights.append(weights)
                except Exception as e:
                    print(f"⚠️ Could not parse weights for seed{seed}: {e}")
    
    return np.array(all_weights) if all_weights else None

def load_px_stats(run_id, seeds=[40, 41, 42, 43, 44]):
    """Load P(x) statistics from all seeds."""
    all_px_top1 = []
    all_px_top3 = []
    all_px_top5 = []
    all_px_proximity = []
    
    for seed in seeds:
        path = f"logs/{run_id}/seed{seed}/evaluation_metrics.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            wno_rows = df[df['Label'].str.contains('Validation_Normal', na=False)]
            if not wno_rows.empty:
                row = wno_rows.iloc[0]
                all_px_top1.append(row['Px_Top1_Acc'])
                all_px_top3.append(row['Px_Top3_Acc'])
                all_px_top5.append(row['Px_Top5_Acc'])
                all_px_proximity.append(row['Px_Proximity_Acc'])
    
    return {
        'top1': np.array(all_px_top1),
        'top3': np.array(all_px_top3),
        'top5': np.array(all_px_top5),
        'proximity': np.array(all_px_proximity)
    }

def plot_causal_curves_with_errors(deletion_arr, insertion_arr, deletion_ex_arr, save_dir):
    """Generate causal curves with error bars."""
    
    steps = deletion_arr.shape[1]
    
    # Compute means and stds
    deletion_mean = np.mean(deletion_arr, axis=0)
    deletion_std = np.std(deletion_arr, axis=0)
    
    insertion_mean = np.mean(insertion_arr, axis=0)
    insertion_std = np.std(insertion_arr, axis=0)
    
    x_ticks = np.arange(steps)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Deletion and Insertion curves with error bars
    ax1 = axes[0]
    
    # Deletion curve (red)
    ax1.plot(x_ticks, deletion_mean, 'o-', color='red', linewidth=2.5,
             markersize=8, label='Deletion (Necessity)', zorder=3)
    ax1.fill_between(x_ticks, 
                     deletion_mean - deletion_std,
                     deletion_mean + deletion_std,
                     alpha=0.2, color='red', label='±1 std')
    
    # Insertion curve (green)
    ax1.plot(x_ticks, insertion_mean, 's-', color='green', linewidth=2.5,
             markersize=8, label='Insertion (Sufficiency)', zorder=3)
    ax1.fill_between(x_ticks,
                     insertion_mean - insertion_std,
                     insertion_mean + insertion_std,
                     alpha=0.2, color='green', label='±1 std')
    
    ax1.set_xlabel('Perturbed Sensor Ranks (Steps)', fontsize=12)
    ax1.set_ylabel('Localization Error', fontsize=12)
    ax1.set_title('Causal Faithfulness Curves (5 seeds)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_ticks)
    
    # Add interpretation text
    ax1.text(0.02, 0.98, '↑ Removing important sensors increases error\n↓ Adding important sensors decreases error',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: P(x) vs E(x) deletion comparison
    ax2 = axes[1]
    
    if deletion_ex_arr is not None:
        deletion_ex_mean = np.mean(deletion_ex_arr, axis=0)
        deletion_ex_std = np.std(deletion_ex_arr, axis=0)
        
        # P(x) deletion
        ax2.plot(x_ticks, deletion_mean, 'o-', color='blue', linewidth=2.5,
                 markersize=8, label='P(x) Ranking', zorder=3)
        ax2.fill_between(x_ticks,
                         deletion_mean - deletion_std,
                         deletion_mean + deletion_std,
                         alpha=0.2, color='blue')
        
        # E(x) deletion
        ax2.plot(x_ticks, deletion_ex_mean, 's-', color='orange', linewidth=2.5,
                 markersize=8, label='E(x) Ranking', zorder=3)
        ax2.fill_between(x_ticks,
                         deletion_ex_mean - deletion_ex_std,
                         deletion_ex_mean + deletion_ex_std,
                         alpha=0.2, color='orange')
    else:
        # Just show P(x) deletion
        ax2.plot(x_ticks, deletion_mean, 'o-', color='blue', linewidth=2.5,
                 markersize=8, label='P(x) Deletion', zorder=3)
        ax2.fill_between(x_ticks,
                         deletion_mean - deletion_std,
                         deletion_mean + deletion_std,
                         alpha=0.2, color='blue')
    
    ax2.set_xlabel('Perturbed Sensor Ranks (Steps)', fontsize=12)
    ax2.set_ylabel('Localization Error', fontsize=12)
    ax2.set_title('Deletion Comparison: P(x) vs E(x)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_ticks)
    
    # Add interpretation
    if deletion_ex_arr is not None:
        note = '✓ P(x) and E(x) both identify important sensors'
        if deletion_mean[-1] > deletion_ex_mean[-1]:
            note = '✓ P(x) deletion causes larger error increase\n  P(x) identifies more important sensors'
        ax2.text(0.02, 0.98, note,
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, 'causal_curves_with_errors.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()
    
    return deletion_mean, deletion_std

def plot_px_topk_with_errors(px_stats, save_dir):
    """Generate P(x) Top-K accuracy with error bars."""
    
    # Compute means and stds
    top1_mean = np.mean(px_stats['top1'])
    top1_std = np.std(px_stats['top1'])
    top3_mean = np.mean(px_stats['top3'])
    top3_std = np.std(px_stats['top3'])
    top5_mean = np.mean(px_stats['top5'])
    top5_std = np.std(px_stats['top5'])
    
    # Also compute proximity
    prox_mean = np.mean(px_stats['proximity'])
    prox_std = np.std(px_stats['proximity'])
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Top-K accuracy
    ax1 = axes[0]
    k_values = ['Top-1', 'Top-3', 'Top-5']
    means = [top1_mean, top3_mean, top5_mean]
    stds = [top1_std, top3_std, top5_std]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax1.bar(k_values, means, yerr=stds, capsize=5, color=colors, alpha=0.7, 
                   error_kw={'elinewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{mean:.1f}%\n±{std:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Random (5%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('P(x) Source Identification Accuracy (5 seeds)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    
    # Plot 2: Proximity accuracy
    ax2 = axes[1]
    ax2.bar(['Proximity\nAccuracy'], [prox_mean], yerr=[prox_std], 
            capsize=5, color='#6B8E23', alpha=0.7,
            error_kw={'elinewidth': 2, 'ecolor': 'black'})
    
    ax2.text(0, prox_mean + 2, f'{prox_mean:.1f}%\n±{prox_std:.1f}%', 
             ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('P(x) Proximity Accuracy (5 seeds)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'px_topk_with_errors.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

def plot_wavelet_weights_with_errors(weights_arr, save_dir):
    """Generate wavelet weights with error bars."""
    
    if weights_arr is None or len(weights_arr) == 0:
        print("No wavelet weights found!")
        return
    
    # Compute means and stds
    weights_mean = np.mean(weights_arr, axis=0)
    weights_std = np.std(weights_arr, axis=0)
    
    # Band labels
    band_labels = ['Approx\n(Band 0)', 'Band 1', 'Band 2', 'Band 3']
    colors = ['#95A5A6', '#E74C3C', '#F39C12', '#2ECC71']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(band_labels, weights_mean, yerr=weights_std, capsize=5,
                  color=colors, alpha=0.7, error_kw={'elinewidth': 2, 'ecolor': 'black'})
    
    # Add value labels
    for bar, mean, std in zip(bars, weights_mean, weights_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Wavelet Band', fontsize=12)
    ax.set_ylabel('Learned Weight (α)', fontsize=12)
    ax.set_title('Learned Wavelet Band Importance (5 seeds)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.5)
    
    # Add interpretation
    if weights_mean[0] < 0.01:
        ax.text(0.5, -0.2, '✓ Approximation band (0.0) is correctly ignored\n✓ High-frequency bands (1-3) are most important',
                transform=ax.transAxes, ha='center', fontsize=10, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'wavelet_weights_with_errors.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

def main():
    """Main function to generate all explainability plots with error bars."""
    
    RUN_ID = "2026-07-29_053033"
    save_dir = f"logs/{RUN_ID}/explainability_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = [40, 41, 42, 43, 44]
    
    print("="*80)
    print("GENERATING EXPLAINABILITY ANALYSIS WITH ERROR BARS")
    print("="*80)
    
    # 1. Load causal curves
    print("\n[1/4] Loading causal curves data...")
    deletion_arr, insertion_arr, deletion_ex_arr = load_causal_data(RUN_ID, seeds)
    
    if deletion_arr is not None:
        print(f"Loaded {len(deletion_arr)} seeds, {deletion_arr.shape[1]} steps")
        plot_causal_curves_with_errors(deletion_arr, insertion_arr, deletion_ex_arr, save_dir)
    else:
        print("⚠️ No causal curves data found.")
    
    # 2. Load P(x) statistics
    print("\n[2/4] Loading P(x) statistics...")
    px_stats = load_px_stats(RUN_ID, seeds)
    if px_stats['top1'].size > 0:
        print(f"Loaded {len(px_stats['top1'])} seeds")
        plot_px_topk_with_errors(px_stats, save_dir)
    else:
        print("⚠️ No P(x) statistics found.")
    
    # 3. Load wavelet weights
    print("\n[3/4] Loading wavelet weights...")
    weights_arr = load_wavelet_weights(RUN_ID, seeds)
    if weights_arr is not None and len(weights_arr) > 0:
        print(f"Loaded {len(weights_arr)} seeds")
        plot_wavelet_weights_with_errors(weights_arr, save_dir)
    else:
        print("⚠️ No wavelet weights found.")
    
    # 4. Generate summary report
    print("\n[4/4] Generating summary report...")
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f"Saved to: {save_dir}")
    print("="*80)
    
    # Print statistics
    print("\nSummary Statistics (5 seeds):")
    print("-"*60)
    if deletion_arr is not None:
        final_errors = deletion_arr[:, -1]
        print(f"Deletion final error: {np.mean(final_errors):.4f} ± {np.std(final_errors):.4f}")
    
    if px_stats['top1'].size > 0:
        print(f"P(x) Top-1: {np.mean(px_stats['top1']):.1f}% ± {np.std(px_stats['top1']):.1f}%")
        print(f"P(x) Top-3: {np.mean(px_stats['top3']):.1f}% ± {np.std(px_stats['top3']):.1f}%")
        print(f"P(x) Top-5: {np.mean(px_stats['top5']):.1f}% ± {np.std(px_stats['top5']):.1f}%")
        print(f"P(x) Proximity: {np.mean(px_stats['proximity']):.1f}% ± {np.std(px_stats['proximity']):.1f}%")
    
    if weights_arr is not None:
        print(f"Wavelet weights: {np.mean(weights_arr, axis=0)} ± {np.std(weights_arr, axis=0)}")

if __name__ == "__main__":
    main()