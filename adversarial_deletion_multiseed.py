# -*- coding: utf-8 -*-
"""
Adversarial Deletion Comparison Across 5 Seeds.
Generates publication-quality figure with error bars.
@author: anjum
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

def load_adversarial_deletion_data(run_id, seeds=[40, 41, 42, 43, 44]):
    """Load adversarial deletion data from all seeds."""
    all_deletion_px = []
    all_deletion_ex = []
    found_seeds = []
    
    for seed in seeds:
        # Look for adversarial causal data
        path = f"logs/{run_id}/seed{seed}/explainability_plots_adversarial/causal_curves_data_Validation_Adversarial.json"
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                all_deletion_px.append(data['deletion_px'])
                all_deletion_ex.append(data.get('deletion_ex', data['deletion_px']))
                found_seeds.append(seed)
        else:
            print(f"⚠️ No adversarial data for seed{seed}")
    
    if not all_deletion_px:
        print("❌ No adversarial deletion data found!")
        return None, None
    
    print(f"✅ Loaded {len(all_deletion_px)} seeds: {found_seeds}")
    
    deletion_px_arr = np.array(all_deletion_px)
    deletion_ex_arr = np.array(all_deletion_ex)
    
    return deletion_px_arr, deletion_ex_arr

def plot_adversarial_deletion_comparison(deletion_px_arr, deletion_ex_arr, save_dir):
    """Generate adversarial deletion comparison with error bars."""
    
    # Compute means and stds
    px_mean = np.mean(deletion_px_arr, axis=0)
    px_std = np.std(deletion_px_arr, axis=0)
    
    ex_mean = np.mean(deletion_ex_arr, axis=0)
    ex_std = np.std(deletion_ex_arr, axis=0)
    
    steps = len(px_mean)
    x_ticks = np.arange(steps)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # P(x) Deletion (Blue)
    ax.plot(x_ticks, px_mean, 'o-', color='#2E86AB', linewidth=2.5,
            markersize=10, label='P(x) Deletion (Adversarial)', zorder=3)
    ax.fill_between(x_ticks,
                     px_mean - px_std,
                     px_mean + px_std,
                     alpha=0.2, color='#2E86AB', label='±1 std (P(x))')
    
    # E(x) Deletion (Red)
    ax.plot(x_ticks, ex_mean, 's-', color='#C73E1D', linewidth=2.5,
            markersize=10, label='E(x) Deletion (Adversarial)', zorder=3)
    ax.fill_between(x_ticks,
                     ex_mean - ex_std,
                     ex_mean + ex_std,
                     alpha=0.2, color='#C73E1D', label='±1 std (E(x))')
    
    # Add baseline reference
    baseline = px_mean[0]
    ax.axhline(y=baseline, color='gray', linestyle=':', alpha=0.5,
               label=f'Baseline: {baseline:.4f}')
    
    # Add final error annotation
    final_px = px_mean[-1]
    final_ex = ex_mean[-1]
    ax.text(steps-1, final_px + 0.02, f'P(x): {final_px:.3f}', 
            ha='right', fontsize=10, color='#2E86AB')
    ax.text(steps-1, final_ex + 0.02, f'E(x): {final_ex:.3f}', 
            ha='right', fontsize=10, color='#C73E1D')
    
    # Labels and title
    ax.set_xlabel('Sensors Removed (Ranked by Importance)', fontsize=12)
    ax.set_ylabel('Localization Error (E(x))', fontsize=12)
    ax.set_title('Adversarial Deletion: P(x) vs E(x) (5 Seeds)', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{i+1}' for i in x_ticks])
    
    # Add interpretation text
    if final_px > final_ex:
        note = '✓ P(x) deletion causes larger error increase\n  P(x) correctly identifies true source despite decoy'
    else:
        note = '✓ P(x) and E(x) both identify important sensors'
    
    ax.text(0.02, 0.98, note,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add statistical annotation
    from scipy import stats
    px_final = deletion_px_arr[:, -1]
    ex_final = deletion_ex_arr[:, -1]
    t_stat, p_value = stats.ttest_rel(px_final, ex_final)
    ax.text(0.02, 0.88, f'p-value: {p_value:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(save_dir, 'adversarial_deletion_comparison_5seeds.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.show()
    plt.close()
    
    return px_mean, px_std, ex_mean, ex_std

def main():
    """Main function."""
    
    RUN_ID = "2026-07-29_053033"
    save_dir = f"logs/{RUN_ID}/explainability_analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = [40, 41, 42, 43, 44]
    
    print("="*80)
    print("ADVERSARIAL DELETION COMPARISON (5 SEEDS)")
    print("="*80)
    
    # Load data
    deletion_px_arr, deletion_ex_arr = load_adversarial_deletion_data(RUN_ID, seeds)
    
    if deletion_px_arr is None:
        print("❌ No adversarial deletion data found!")
        print("Run the adversarial evaluation first.")
        return
    
    print(f"\nDeletion P(x) shape: {deletion_px_arr.shape}")
    print(f"Deletion E(x) shape: {deletion_ex_arr.shape}")
    
    # Generate plot
    px_mean, px_std, ex_mean, ex_std = plot_adversarial_deletion_comparison(
        deletion_px_arr, deletion_ex_arr, save_dir
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nP(x) Deletion (Adversarial):")
    print(f"  Final error: {px_mean[-1]:.4f} ± {px_std[-1]:.4f}")
    print(f"  Error increase: {(px_mean[-1] - px_mean[0]) / px_mean[0] * 100:.1f}%")
    
    print(f"\nE(x) Deletion (Adversarial):")
    print(f"  Final error: {ex_mean[-1]:.4f} ± {ex_std[-1]:.4f}")
    print(f"  Error increase: {(ex_mean[-1] - ex_mean[0]) / ex_mean[0] * 100:.1f}%")
    
    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(deletion_px_arr[:, -1], deletion_ex_arr[:, -1])
    print(f"\nPaired t-test (P(x) vs E(x) final errors):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  ✓ Statistically significant difference!")
    else:
        print("  ✗ Not statistically significant (tied)")
    
    print(f"\n✅ Figure saved to: {save_dir}/adversarial_deletion_comparison_5seeds.png")

if __name__ == "__main__":
    main()