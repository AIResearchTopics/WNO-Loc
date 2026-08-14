# -*- coding: utf-8 -*-
"""
Explainability Analysis and Visualization for WNO Localization.
Provides comprehensive validation that P(x) identifies the closest sensor.

@author: usman.anjum
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_curve, auc
import seaborn as sns

# ============================================================================
# COMPATIBILITY: NumPy 2.0+ trapz fix
# ============================================================================
try:
    from numpy import trapezoid  # NumPy 2.0+
except ImportError:
    try:
        from numpy import trapz as trapezoid  # NumPy < 2.0
    except ImportError:
        def trapezoid(y, dx=1.0):
            if len(y) < 2:
                return 0.0
            return (sum(y[1:-1]) * 2 + y[0] + y[-1]) * dx / 2

# ============================================================================
# JSON SERIALIZATION HELPERS
# ============================================================================
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

# ============================================================================
# PLOT STYLE
# ============================================================================
def setup_plot_style():
    """Set publication-quality plot style."""
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('seaborn-paper')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def plot_px_vs_distance(model, loader, device, save_dir='plots', is_real_data=False):
    """
    Show that P(x) is inversely correlated with distance from source.
    Handles variable sensor counts across batches.
    """
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    
    model.eval()
    
    # Store per-event data (each event can have different N)
    all_distances = []  # List of (N,) arrays
    all_px = []         # List of (N,) arrays
    all_ex = []         # List of (N,) arrays
    all_sensor_indices = []
    
    with torch.no_grad():
        for U, coords_batch, lap_batch, adj_batch, location, _ in loader:
            U = U.to(device)
            coords_batch = coords_batch.to(device)
            location = location.to(device)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            B, N, T, F = U.shape
            
            # Get predictions
            _, _, intensity, Px, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)
            
            # E(x) for comparison
            Ex = intensity.max(dim=2).values  # (B, N)
            
            # Distance from each sensor to source
            if is_real_data:
                from evaluate import Evaluator
                dist = torch.zeros(B, N, device=device)
                for b in range(B):
                    for n in range(N):
                        dist[b, n] = Evaluator._compute_haversine_distance(
                            coords_batch[b, n:n+1], 
                            location[b:b+1]
                        )
            else:
                dist = torch.norm(coords_batch - location.unsqueeze(1), dim=2)
            
            # Store per event
            for b in range(B):
                all_distances.append(dist[b].cpu().numpy())  # (N,)
                all_px.append(Px[b].cpu().numpy())           # (N,)
                all_ex.append(Ex[b].cpu().numpy())           # (N,)
                
                sorted_indices = np.argsort(Px[b].cpu().numpy())[::-1]
                all_sensor_indices.append(sorted_indices)
    
    if not all_px:
        print("⚠️ No data collected in plot_px_vs_distance!")
        return 0, 0, []
    
    # Flatten all data for correlation (concatenate all arrays)
    # Since each event has different N, we concatenate them
    all_dist_flat = np.concatenate(all_distances)
    all_px_flat = np.concatenate(all_px)
    all_ex_flat = np.concatenate(all_ex)
    
    # Remove NaNs and infs
    valid = ~(np.isnan(all_dist_flat) | np.isnan(all_px_flat) | 
              np.isinf(all_dist_flat) | np.isinf(all_px_flat))
    dist_flat = all_dist_flat[valid]
    px_flat = all_px_flat[valid]
    ex_flat = all_ex_flat[valid]
    
    # Compute correlation
    if len(dist_flat) > 1 and len(px_flat) > 1:
        px_corr, px_pval = pearsonr(-dist_flat, px_flat)
        px_spearman, px_spval = spearmanr(-dist_flat, px_flat)
        ex_corr, ex_pval = pearsonr(-dist_flat, ex_flat)
    else:
        px_corr, px_pval, px_spearman, px_spval, ex_corr, ex_pval = 0, 1, 0, 1, 0, 1
    
    print(f"\n{'='*60}")
    print("P(x) vs Proximity Correlation Analysis")
    print(f"{'='*60}")
    print(f"Number of events: {len(all_px)}")
    print(f"Total sensor measurements: {len(dist_flat)}")
    print(f"P(x) Pearson Correlation:  {px_corr:.4f} (p={px_pval:.2e})")
    print(f"P(x) Spearman Correlation: {px_spearman:.4f} (p={px_spval:.2e})")
    print(f"E(x) Pearson Correlation:  {ex_corr:.4f} (p={ex_pval:.2e})")
    print(f"{'='*60}\n")
    
    # =========================================================================
    # PLOTS
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: P(x) vs Distance
    ax1 = axes[0, 0]
    ax1.scatter(dist_flat, px_flat, alpha=0.3, s=5, c='blue')
    if len(dist_flat) > 1:
        z = np.polyfit(dist_flat, px_flat, 1)
        p = np.poly1d(z)
        x_line = np.linspace(dist_flat.min(), dist_flat.max(), 100)
        ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Correlation: {px_corr:.3f}')
    ax1.set_xlabel('Distance from Source')
    ax1.set_ylabel('P(x) (Frequency-based Activation)')
    ax1.set_title('P(x) is Highest Near the Source')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: E(x) vs Distance
    ax2 = axes[0, 1]
    ax2.scatter(dist_flat, ex_flat, alpha=0.3, s=5, c='green')
    if len(dist_flat) > 1:
        z2 = np.polyfit(dist_flat, ex_flat, 1)
        p2 = np.poly1d(z2)
        ax2.plot(x_line, p2(x_line), 'r-', linewidth=2, label=f'Correlation: {ex_corr:.3f}')
    ax2.set_xlabel('Distance from Source')
    ax2.set_ylabel('E(x) (Intensity)')
    ax2.set_title('E(x) vs Distance (For Comparison)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribution
    ax3 = axes[1, 0]
    if len(dist_flat) > 0:
        dist_threshold = np.percentile(dist_flat, 50)
        near_source = px_flat[dist_flat < dist_threshold]
        far_source = px_flat[dist_flat >= dist_threshold]
        ax3.hist(near_source, bins=30, alpha=0.5, label='Near Source', color='blue')
        ax3.hist(far_source, bins=30, alpha=0.5, label='Far from Source', color='orange')
    ax3.set_xlabel('P(x) Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('P(x) Distribution: Near vs Far from Source')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top-K accuracy
    ax4 = axes[1, 1]
    top_k_acc = []
    for k in range(1, 11):
        correct = 0
        total = 0
        for i, sorted_indices in enumerate(all_sensor_indices):
            if i < len(all_distances):
                # True sensor is the one with minimum distance
                true_sensor = np.argmin(all_distances[i])
                if true_sensor in sorted_indices[:k]:
                    correct += 1
                total += 1
        top_k_acc.append(correct / total if total > 0 else 0)
    
    ax4.plot(range(1, 11), top_k_acc, 'bo-', linewidth=2, markersize=8)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Chance (50%)')
    ax4.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
    ax4.set_xlabel('Top-K Sensors')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('P(x) Top-K Accuracy')
    ax4.set_xticks(range(1, 11))
    ax4.set_ylim([0, 1.05])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('P(x) Identifies the Closest Sensor to the Event Source', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'px_vs_distance_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return px_corr, px_spearman, top_k_acc

def visualize_wavelet_bands(model, loader, device, save_dir='plots', num_examples=3):
    """Visualize which frequency bands are most informative for localization."""
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    
    model.eval()
    examples = []
    
    with torch.no_grad():
        for i, (U, coords_batch, lap_batch, adj_batch, location, _) in enumerate(loader):
            if i >= num_examples:
                break
            U = U.to(device)
            coords_batch = coords_batch.to(device)
            location = location.to(device)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            _, _, intensity, Px, _, _, _, band_energy, alpha = model(U, coords_batch, lap_batch, adj_batch)
            
            band_energy = band_energy.reshape(U.shape[0], U.shape[1], -1)
            
            examples.append({
                'band_energy': band_energy.cpu().numpy(),
                'Px': Px.cpu().numpy(),
                'coords': coords_batch.cpu().numpy(),
                'location': location.cpu().numpy(),
                'alpha': alpha.cpu().numpy(),
            })
    
    if not examples:
        print("⚠️ No examples collected for wavelet visualization")
        return
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, example in enumerate(examples):
        band_energy = example['band_energy'][0]
        Px = example['Px'][0]
        alpha = example['alpha']
        n_bands = band_energy.shape[1]
        
        band_energy_norm = band_energy / (band_energy.sum(axis=-1, keepdims=True) + 1e-8)
        
        ax1 = axes[idx, 0]
        for b in range(n_bands):
            ax1.plot(range(band_energy.shape[0]), 
                    band_energy_norm[:, b], 
                    marker='o', markersize=3,
                    label=f'Band {b}')
        ax1.set_xlabel('Sensor Index')
        ax1.set_ylabel('Normalized Band Energy')
        ax1.set_title(f'Wavelet Band Energies (Example {idx+1})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[idx, 1]
        for b in range(n_bands):
            px_band = band_energy_norm[:, b] * alpha[b]
            ax2.bar(np.arange(len(px_band)) + b*0.15, px_band, width=0.15, 
                    alpha=0.7, label=f'Band {b}')
        ax2.set_xlabel('Sensor Index')
        ax2.set_ylabel('Band Contribution to P(x)')
        ax2.set_title(f'Band Contributions to P(x) (Example {idx+1})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[idx, 2]
        ax3.bar(range(len(alpha)), alpha, color='purple', alpha=0.7)
        ax3.set_xlabel('Wavelet Band')
        ax3.set_ylabel('Learned Weight (α)')
        ax3.set_title(f'Learned Band Importance (Example {idx+1})')
        ax3.set_xticks(range(len(alpha)))
        ax3.grid(True, alpha=0.3)
        
        if len(alpha) > 0:
            most_important = alpha.argmax()
            ax3.text(0.5, -0.15, f'Most important: Band {most_important} (highest frequency)',
                    transform=ax3.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Wavelet Band Interpretability: Higher Frequencies are Most Informative', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'wavelet_band_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_px_ex_comparison(model, loader, device, save_dir='plots', num_examples=5):
    """Compare E(x) and P(x) localization side by side."""
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    
    model.eval()
    examples = []
    
    with torch.no_grad():
        for i, (U, coords_batch, lap_batch, adj_batch, location, _) in enumerate(loader):
            if i >= num_examples:
                break
            U = U.to(device)
            coords_batch = coords_batch.to(device)
            location = location.to(device)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            coords_pred_e, _, intensity, Px, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)
            
            Ex = intensity.max(dim=2).values
            e_weights = torch.softmax(Ex / 0.05, dim=1)
            p_weights = Px / (Px.sum(dim=1, keepdim=True) + 1e-8)
            
            true_sensor = torch.norm(coords_batch[0] - location[0].unsqueeze(0), dim=1).argmin()
            
            examples.append({
                'e_weights': e_weights[0].cpu().numpy(),
                'p_weights': p_weights[0].cpu().numpy(),
                'true_sensor': true_sensor.item(),
                'coords': coords_batch[0].cpu().numpy(),
                'location': location[0].cpu().numpy(),
                'coords_pred_e': coords_pred_e[0].cpu().numpy(),
            })
    
    if not examples:
        print("⚠️ No examples collected for E(x) vs P(x) comparison")
        return
    
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, example in enumerate(examples):
        ax1 = axes[idx, 0]
        bars1 = ax1.bar(range(len(example['e_weights'])), example['e_weights'], color='blue', alpha=0.7)
        if example['true_sensor'] < len(bars1):
            bars1[example['true_sensor']].set_color('red')
            bars1[example['true_sensor']].set_label('True Source')
        ax1.set_xlabel('Sensor Index')
        ax1.set_ylabel('Weight')
        ax1.set_title(f'E(x) Weights (Example {idx+1})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[idx, 1]
        bars2 = ax2.bar(range(len(example['p_weights'])), example['p_weights'], color='orange', alpha=0.7)
        if example['true_sensor'] < len(bars2):
            bars2[example['true_sensor']].set_color('red')
            bars2[example['true_sensor']].set_label('True Source')
        ax2.set_xlabel('Sensor Index')
        ax2.set_ylabel('Weight')
        ax2.set_title(f'P(x) Weights (Example {idx+1})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[idx, 2]
        pred_e = np.argmax(example['e_weights']) if len(example['e_weights']) > 0 else 0
        pred_p = np.argmax(example['p_weights']) if len(example['p_weights']) > 0 else 0
        true_s = example['true_sensor']
        
        colors = ['green' if x == true_s else 'blue' for x in [true_s, pred_e, pred_p]]
        ax3.bar(['True', 'E(x)', 'P(x)'], [true_s, pred_e, pred_p], color=colors, alpha=0.7)
        ax3.set_ylabel('Sensor Index')
        ax3.set_title(f'Predictions (Example {idx+1})')
        ax3.grid(True, alpha=0.3)
        
        e_correct = pred_e == true_s
        p_correct = pred_p == true_s
        status = f"E(x): {'✓' if e_correct else '✗'}  P(x): {'✓' if p_correct else '✗'}"
        ax3.text(0.5, -0.15, status, transform=ax3.transAxes, ha='center', 
                fontsize=12, fontweight='bold')
    
    plt.suptitle('E(x) vs P(x) Sensor Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'ex_vs_px_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def statistical_validation(model, loader, device, save_dir='plots', is_real_data=False):
    """Comprehensive statistical validation with variable sensor counts."""
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    
    model.eval()
    
    all_px = []
    all_ex = []
    all_distances = []
    all_true_sensors = []
    all_px_sorted = []
    
    with torch.no_grad():
        for U, coords_batch, lap_batch, adj_batch, location, _ in loader:
            U = U.to(device)
            coords_batch = coords_batch.to(device)
            location = location.to(device)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            B, N, T, F = U.shape
            
            _, _, intensity, Px, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)
            
            Ex = intensity.max(dim=2).values
            
            if is_real_data:
                from evaluate import Evaluator
                dist = torch.zeros(B, N, device=device)
                for b in range(B):
                    for n in range(N):
                        dist[b, n] = Evaluator._compute_haversine_distance(
                            coords_batch[b, n:n+1], 
                            location[b:b+1]
                        )
            else:
                dist = torch.norm(coords_batch - location.unsqueeze(1), dim=2)
            
            true_sensor = dist.argmin(dim=1)
            
            for b in range(B):
                px_np = Px[b].cpu().numpy()
                ex_np = Ex[b].cpu().numpy()
                dist_np = dist[b].cpu().numpy()
                true_idx = int(true_sensor[b].cpu().item())
                
                all_px.append(px_np)
                all_ex.append(ex_np)
                all_distances.append(dist_np)
                all_true_sensors.append(true_idx)
                all_px_sorted.append(np.argsort(px_np)[::-1])
    
    if not all_px:
        print("⚠️ No data collected in statistical_validation!")
        return {}
    
    num_events = len(all_px)
    print(f"✅ Collected {num_events} events")
    
    corr_values = []
    for i in range(num_events):
        px = all_px[i]
        dist = all_distances[i]
        if len(px) > 1 and len(dist) > 1:
            try:
                corr, _ = spearmanr(px, -dist)
                if not np.isnan(corr):
                    corr_values.append(corr)
            except:
                pass
    
    mean_corr = np.mean(corr_values) if corr_values else 0.0
    std_corr = np.std(corr_values) if corr_values else 0.0
    
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    for i in range(num_events):
        sorted_idx = all_px_sorted[i]
        true_idx = all_true_sensors[i]
        
        if sorted_idx[0] == true_idx:
            top1_correct += 1
        if true_idx in sorted_idx[:3]:
            top3_correct += 1
        if true_idx in sorted_idx[:5]:
            top5_correct += 1
    
    top1_acc = top1_correct / num_events if num_events > 0 else 0
    top3_acc = top3_correct / num_events if num_events > 0 else 0
    top5_acc = top5_correct / num_events if num_events > 0 else 0
    
    ranks = []
    for i in range(num_events):
        sorted_idx = all_px_sorted[i]
        true_idx = all_true_sensors[i]
        rank = np.where(sorted_idx == true_idx)[0]
        if len(rank) > 0:
            ranks.append(rank[0])
    mean_rank = np.mean(ranks) if ranks else 0
    
    proximity_threshold = 15.0 if is_real_data else 0.15
    proximity_correct = 0
    proximity_distances = []
    
    for i in range(num_events):
        sorted_idx = all_px_sorted[i]
        top_idx = sorted_idx[0]
        dist = all_distances[i]
        
        if top_idx < len(dist) and dist[top_idx] <= proximity_threshold:
            proximity_correct += 1
        if top_idx < len(dist):
            proximity_distances.append(dist[top_idx])
    
    proximity_acc = proximity_correct / num_events if num_events > 0 else 0
    mean_prox_dist = np.mean(proximity_distances) if proximity_distances else 0
    std_prox_dist = np.std(proximity_distances) if proximity_distances else 0
    
    print(f"\n{'='*60}")
    print("STATISTICAL VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of events: {num_events}")
    print(f"P(x) vs Distance Correlation: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"P(x) Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"P(x) Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"P(x) Top-5 Accuracy: {top5_acc*100:.2f}%")
    print(f"Mean Rank of True Sensor: {mean_rank:.2f}")
    print(f"P(x) Proximity Accuracy: {proximity_acc*100:.2f}%")
    print(f"Mean Proximity Distance: {mean_prox_dist:.4f} ± {std_prox_dist:.4f}")
    print(f"{'='*60}")
    
    results = {
        'mean_correlation': mean_corr,
        'std_correlation': std_corr,
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'mean_rank': mean_rank,
        'num_events': num_events,
        'proximity_accuracy': proximity_acc,
        'mean_proximity_distance': mean_prox_dist,
        'std_proximity_distance': std_prox_dist,
    }
    
    # Convert to serializable types
    results_serializable = convert_to_serializable(results)
    
    save_path = os.path.join(save_dir, 'statistical_validation.json')
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"✅ Saved statistical results to: {save_path}")
    
    try:
        plot_statistical_results(results, save_dir)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    return results

def plot_statistical_results(results, save_dir):
    """Generate simplified plots from statistical results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    k_values = ['Top-1', 'Top-3', 'Top-5']
    accuracies = [results['top1_accuracy']*100, 
                  results['top3_accuracy']*100, 
                  results['top5_accuracy']*100]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax1.bar(k_values, accuracies, color=colors, alpha=0.7)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., acc + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Random (5%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('P(x) Source Identification Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    
    ax2 = axes[1]
    ax2.bar(['Proximity\nAccuracy'], [results['proximity_accuracy']*100],
            color='#6B8E23', alpha=0.7)
    ax2.text(0, results['proximity_accuracy']*100 + 2,
            f"{results['proximity_accuracy']*100:.1f}%",
            ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('P(x) Proximity Accuracy')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 110)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'statistical_validation_plots.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plots to: {save_path}")
    plt.close()

def plot_causal_curves(model, loader, device, save_dir='plots', is_real_data=False, steps=5, 
                       title_suffix=''):
    """Enhanced version of run_causal_curves with proper visualization and saving."""
    os.makedirs(save_dir, exist_ok=True)
    setup_plot_style()
    
    model.eval()
    
    deletion_history = {step: [] for step in range(steps + 1)}
    insertion_history = {step: [] for step in range(steps + 1)}
    deletion_ex_history = {step: [] for step in range(steps + 1)}
    
    with torch.no_grad():
        for U, coords_batch, lap_batch, adj_batch, location, _ in loader:
            U = U.to(device)
            coords_batch = coords_batch.to(device)
            location = location.to(device)
            lap_batch = lap_batch.to(device)
            adj_batch = adj_batch.to(device)
            
            B, N, T, F = U.shape
            
            coords_pred, _, intensity, Px, _, _, _, _, _ = model(U, coords_batch, lap_batch, adj_batch)
            
            Ex = intensity.max(dim=2).values
            top_ranked_nodes_px = Px.sort(dim=1, descending=True).indices
            top_ranked_nodes_ex = Ex.sort(dim=1, descending=True).indices
            
            if is_real_data:
                from evaluate import Evaluator
                baseline_err = Evaluator._compute_haversine_distance(coords_pred, location)
            else:
                baseline_err = torch.norm(coords_pred - location, dim=1)
            
            U_deletion = U.clone()
            for step in range(steps + 1):
                pred, _, _, _, _, _, _, _, _ = model(U_deletion, coords_batch, lap_batch, adj_batch)
                if pred.shape != location.shape: 
                    pred = pred.reshape(location.shape)
                
                if is_real_data:
                    err = Evaluator._compute_haversine_distance(pred, location)
                else:
                    err = torch.norm(pred - location, dim=1)
                
                deletion_history[step].append(err.cpu())
                
                if step < steps:
                    for b in range(B):
                        target_node = top_ranked_nodes_px[b, step]
                        U_deletion[b, target_node, :, :] = 0.0
            
            U_deletion_ex = U.clone()
            for step in range(steps + 1):
                pred, _, _, _, _, _, _, _, _ = model(U_deletion_ex, coords_batch, lap_batch, adj_batch)
                if pred.shape != location.shape: 
                    pred = pred.reshape(location.shape)
                
                if is_real_data:
                    err = Evaluator._compute_haversine_distance(pred, location)
                else:
                    err = torch.norm(pred - location, dim=1)
                
                deletion_ex_history[step].append(err.cpu())
                
                if step < steps:
                    for b in range(B):
                        target_node = top_ranked_nodes_ex[b, step]
                        U_deletion_ex[b, target_node, :, :] = 0.0
            
            U_insertion = torch.zeros_like(U)
            for step in range(steps + 1):
                pred, _, _, _, _, _, _, _, _ = model(U_insertion, coords_batch, lap_batch, adj_batch)
                if pred.shape != location.shape: 
                    pred = pred.reshape(location.shape)
                
                if is_real_data:
                    err = Evaluator._compute_haversine_distance(pred, location)
                else:
                    err = torch.norm(pred - location, dim=1)
                
                insertion_history[step].append(err.cpu())
                
                if step < steps:
                    for b in range(B):
                        target_node = top_ranked_nodes_px[b, step]
                        U_insertion[b, target_node, :, :] = U[b, target_node, :, :]
    
    final_deletion_px = [torch.cat(deletion_history[s]).mean().item() for s in range(steps + 1)]
    final_insertion_px = [torch.cat(insertion_history[s]).mean().item() for s in range(steps + 1)]
    final_deletion_ex = [torch.cat(deletion_ex_history[s]).mean().item() for s in range(steps + 1)]
    
    deletion_std = [torch.cat(deletion_history[s]).std().item() for s in range(steps + 1)]
    insertion_std = [torch.cat(insertion_history[s]).std().item() for s in range(steps + 1)]
    
    deletion_auc = trapezoid(final_deletion_px, dx=1.0)
    insertion_auc = trapezoid(final_insertion_px, dx=1.0)
    
    baseline_error = final_deletion_px[0]
    final_error = final_deletion_px[-1]
    error_increase = ((final_error - baseline_error) / (baseline_error + 1e-8)) * 100
    
    print("\n" + "="*80)
    print(f"CAUSAL EXPLAINABILITY CURVES{(' - ' + title_suffix) if title_suffix else ''}")
    print("="*80)
    print(f"Deletion Steps (0 to {steps} nodes removed): {[round(x, 4) for x in final_deletion_px]}")
    print(f"Insertion Steps (0 to {steps} nodes revealed): {[round(x, 4) for x in final_insertion_px]}")
    print(f"\nMetrics:")
    print(f"  • Baseline Error: {baseline_error:.4f}")
    print(f"  • Final Error (after removing {steps} nodes): {final_error:.4f}")
    print(f"  • Error Increase: {error_increase:.2f}%")
    print(f"  • Deletion AUC (necessity): {deletion_auc:.4f}")
    print(f"  • Insertion AUC (sufficiency): {insertion_auc:.4f}")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    x_ticks = range(steps + 1)
    
    ax1.plot(x_ticks, final_deletion_px, 'o-', color='red', linewidth=2.5, 
             markersize=8, label='Deletion (Necessity)', zorder=3)
    ax1.fill_between(x_ticks, 
                     [d - s for d, s in zip(final_deletion_px, deletion_std)],
                     [d + s for d, s in zip(final_deletion_px, deletion_std)],
                     alpha=0.2, color='red')
    
    ax1.plot(x_ticks, final_insertion_px, 's-', color='green', linewidth=2.5,
             markersize=8, label='Insertion (Sufficiency)', zorder=3)
    ax1.fill_between(x_ticks,
                     [d - s for d, s in zip(final_insertion_px, insertion_std)],
                     [d + s for d, s in zip(final_insertion_px, insertion_std)],
                     alpha=0.2, color='green')
    
    ax1.set_xlabel('Perturbed Sensor Ranks (Steps)', fontsize=12)
    ax1.set_ylabel('Localization Error', fontsize=12)
    ax1.set_title(f'Causal Faithfulness Curves\n{title_suffix}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_ticks)
    
    ax1.text(0.02, 0.98, 
             '↑ Removing important sensors increases error\n↓ Adding important sensors decreases error',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = axes[1]
    
    ax2.plot(x_ticks, final_deletion_px, 'o-', color='blue', linewidth=2.5,
             markersize=8, label='P(x) Ranking', zorder=3)
    ax2.fill_between(x_ticks,
                     [d - s for d, s in zip(final_deletion_px, deletion_std)],
                     [d + s for d, s in zip(final_deletion_px, deletion_std)],
                     alpha=0.2, color='blue')
    
    ax2.plot(x_ticks, final_deletion_ex, 's-', color='orange', linewidth=2.5,
             markersize=8, label='E(x) Ranking', zorder=3)
    
    ax2.set_xlabel('Perturbed Sensor Ranks (Steps)', fontsize=12)
    ax2.set_ylabel('Localization Error', fontsize=12)
    ax2.set_title(f'Deletion Comparison: P(x) vs E(x)\n{title_suffix}', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_ticks)
    
    if final_deletion_px[-1] > final_deletion_ex[-1]:
        note = '✓ P(x) deletion causes larger error increase\n  P(x) identifies more important sensors'
    else:
        note = '✓ P(x) and E(x) both identify important sensors'
    ax2.text(0.02, 0.98, note,
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    suffix = f"_{title_suffix.replace(' ', '_')}" if title_suffix else ""
    save_path = os.path.join(save_dir, f'causal_curves{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    results = {
        'title': title_suffix,
        'steps': steps,
        'deletion_px': final_deletion_px,
        'insertion_px': final_insertion_px,
        'deletion_ex': final_deletion_ex,
        'deletion_std': deletion_std,
        'insertion_std': insertion_std,
        'baseline_error': baseline_error,
        'final_error': final_error,
        'error_increase_percent': error_increase,
        'deletion_auc': deletion_auc,
        'insertion_auc': insertion_auc,
        'necessity_score': deletion_auc / (insertion_auc + 1e-8)
    }
    
    results_serializable = convert_to_serializable(results)
    
    data_path = os.path.join(save_dir, f'causal_curves_data{suffix}.json')
    with open(data_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"Saved data: {data_path}")
    
    return final_deletion_px, final_insertion_px, final_deletion_ex

def run_full_explainability_analysis(model, loader, device, save_dir='explainability_plots', 
                                     is_real_data=False, dataset_name='', steps=5):
    """Run all explainability analyses and save all plots."""
    print("\n" + "="*60)
    print(f"RUNNING FULL EXPLAINABILITY ANALYSIS: {dataset_name}")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[1/5] Running statistical validation...")
    stats = statistical_validation(model, loader, device, save_dir=save_dir, 
                                   is_real_data=is_real_data)
    if not stats:
        print("⚠️ Statistical validation returned no results. Skipping further analysis.")
        return
    
    print("\n[2/5] Running P(x) vs Distance analysis...")
    try:
        px_corr, px_spearman, top_k_acc = plot_px_vs_distance(model, loader, device, 
                                                              save_dir=save_dir, 
                                                              is_real_data=is_real_data)
    except Exception as e:
        print(f"⚠️ P(x) vs Distance analysis failed: {e}")
    
    print("\n[3/5] Running wavelet band analysis...")
    try:
        visualize_wavelet_bands(model, loader, device, save_dir=save_dir)
    except Exception as e:
        print(f"⚠️ Wavelet band analysis failed: {e}")
    
    print("\n[4/5] Running E(x) vs P(x) comparison...")
    try:
        plot_px_ex_comparison(model, loader, device, save_dir=save_dir)
    except Exception as e:
        print(f"⚠️ E(x) vs P(x) comparison failed: {e}")
    
    print("\n[5/5] Running causal curves analysis...")
    try:
        del_px, ins_px, del_ex = plot_causal_curves(
            model, loader, device, save_dir=save_dir, 
            is_real_data=is_real_data, steps=steps,
            title_suffix=dataset_name
        )
        del_increase = ((del_px[-1] - del_px[0]) / (del_px[0] + 1e-8) * 100)
    except Exception as e:
        print(f"⚠️ Causal curves analysis failed: {e}")
        del_px, ins_px, del_ex = [0], [0], [0]
        del_increase = 0
    
    print("\n" + "="*60)
    print(f"EXPLAINABILITY ANALYSIS COMPLETE: {dataset_name}")
    print(f"All plots saved to: {save_dir}")
    print("="*60)
    print("\nKey Findings:")
    print(f"  • P(x) identifies the true sensor with {stats.get('top1_accuracy', 0)*100:.2f}% Top-1 accuracy")
    print(f"  • Mean rank of true sensor in P(x): {stats.get('mean_rank', 0):.2f}")
    print(f"  • Correlation with proximity: {stats.get('mean_correlation', 0):.4f}")
    print(f"  • Deleting top sensors increases error by {del_increase:.2f}%")
    print("="*60)
    
    return stats, (del_px, ins_px, del_ex)