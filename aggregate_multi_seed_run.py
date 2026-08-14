# -*- coding: utf-8 -*-
"""
Multi-Seed Results Aggregator with Statistical Tests.
Automatically finds and analyzes all seed results.
@author: anjum
"""

import pandas as pd
import numpy as np
import glob
import os
from scipy import stats
from scipy.stats import ttest_rel, ttest_ind
import re
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

LOGS_PATH = 'logs'

# ============================================================================
# FUNCTIONS
# ============================================================================

def list_all_runs():
    """List all available RUN_IDs in logs directory."""
    if not os.path.exists(LOGS_PATH):
        print(f"Logs directory not found: {LOGS_PATH}")
        return []
    
    log_dirs = glob.glob(f"{LOGS_PATH}/*")
    
    if not log_dirs:
        print("No log directories found!")
        return []
    
    run_list = []
    for d in log_dirs:
        run_id = os.path.basename(d)
        
        seeds = glob.glob(f"{d}/seed*")
        num_seeds = len(seeds)
        
        has_results = False
        for seed_dir in seeds:
            csv_path = os.path.join(seed_dir, "evaluation_metrics.csv")
            if os.path.exists(csv_path):
                has_results = True
                break
        
        try:
            if '_' in run_id:
                dt = datetime.strptime(run_id, '%Y-%m-%d_%H%M%S')
            else:
                dt = datetime.fromtimestamp(os.path.getmtime(d))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = "Unknown"
        
        run_list.append({
            'run_id': run_id,
            'num_seeds': num_seeds,
            'has_results': has_results,
            'time': time_str,
            'path': d
        })
    
    return run_list

def display_runs_table(run_list):
    """Display available runs in a nice table format."""
    if not run_list:
        print("No runs found!")
        return
    
    print("\n" + "="*80)
    print("AVAILABLE RUNS")
    print("="*80)
    print(f"{'#':<3} {'RUN_ID':<25} {'Seeds':<6} {'Results':<8} {'Time':<20}")
    print("-"*80)
    
    for i, run in enumerate(run_list):
        status = "✅" if run['has_results'] else "❌"
        print(f"{i:<3} {run['run_id']:<25} {run['num_seeds']:<6} {status:<8} {run['time']:<20}")
    
    print("="*80)

def select_run_interactive(run_list):
    """Let user select a run interactively."""
    if not run_list:
        return None
    
    valid_runs = [r for r in run_list if r['has_results']]
    
    if not valid_runs:
        print("No runs with complete results found!")
        return None
    
    display_runs_table(valid_runs)
    
    if len(valid_runs) == 1:
        print(f"\nOnly one valid run found. Using: {valid_runs[0]['run_id']}")
        return valid_runs[0]['run_id']
    
    while True:
        try:
            choice = input("\nSelect a RUN_ID (enter number, or 'latest'): ").strip()
            
            if choice.lower() == 'latest':
                valid_runs.sort(key=lambda x: x['time'], reverse=True)
                return valid_runs[0]['run_id']
            
            idx = int(choice)
            if 0 <= idx < len(valid_runs):
                return valid_runs[idx]['run_id']
            else:
                print(f"Invalid choice. Enter 0-{len(valid_runs)-1}")
        except ValueError:
            print("Invalid input. Enter a number or 'latest'")

def load_all_seeds(run_id):
    """Load all seed results for a given RUN_ID."""
    run_path = os.path.join(LOGS_PATH, run_id)
    files = glob.glob(f"{run_path}/seed*/evaluation_metrics.csv")
    
    if not files:
        print(f"Warning: No files found for RUN_ID={run_id}")
        return None
    
    print(f"\nFound {len(files)} seed runs for RUN_ID={run_id}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            seed_match = re.search(r'seed(\d+)', f)
            if seed_match:
                df['seed'] = int(seed_match.group(1))
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
    
    if not dfs:
        return None
    
    all_runs = pd.concat(dfs, ignore_index=True)
    return all_runs

def get_model_type(label):
    """Detect model type from label dynamically."""
    if 'BiLSTM' in label:
        return 'BiLSTM'
    elif 'Transformer' in label:
        return 'Transformer'
    elif 'FNO' in label:
        return 'FNO'
    elif 'IDW' in label:
        return 'IDW'
    else:
        # Everything else is WNO (our model)
        return 'WNO'

def create_summary_table(all_runs, metrics_of_interest, run_id):
    """Create comprehensive summary table with means and stds."""
    
    summary = all_runs.groupby('Label')[metrics_of_interest].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    metric_cols = []
    for metric in metrics_of_interest:
        metric_cols.append(f'{metric}_mean')
        metric_cols.append(f'{metric}_std')
    
    count_cols = [col for col in summary.columns if 'count' in col]
    selected_cols = metric_cols + count_cols if count_cols else metric_cols
    summary = summary[selected_cols]
    
    summary['Model_Type'] = summary.index.map(get_model_type)
    
    cols = ['Model_Type'] + [col for col in summary.columns if col != 'Model_Type']
    summary = summary[cols]
    
    os.makedirs(f"{LOGS_PATH}/{run_id}/analysis", exist_ok=True)
    summary.to_csv(f"{LOGS_PATH}/{run_id}/analysis/multiseed_summary.csv")
    print(f"\nSummary saved to: {LOGS_PATH}/{run_id}/analysis/multiseed_summary.csv")
    
    return summary

def compute_statistical_tests(all_runs, metrics_of_interest):
    """Compute p-values and effect sizes between WNO and baselines."""
    results = {}
    
    labels = all_runs['Label'].unique()
    
    wno_labels = [l for l in labels if get_model_type(l) == 'WNO']
    lstm_labels = [l for l in labels if get_model_type(l) == 'BiLSTM']
    trans_labels = [l for l in labels if get_model_type(l) == 'Transformer']
    fno_labels = [l for l in labels if get_model_type(l) == 'FNO']
    
    wno_label = wno_labels[0] if wno_labels else None
    lstm_label = lstm_labels[0] if lstm_labels else None
    trans_label = trans_labels[0] if trans_labels else None
    fno_label = fno_labels[0] if fno_labels else None
    
    if not wno_label:
        print("Warning: No WNO label found!")
        return results
    
    print(f"\nComparing models:")
    print(f"  WNO: {wno_label}")
    if lstm_label: print(f"  BiLSTM: {lstm_label}")
    if trans_label: print(f"  Transformer: {trans_label}")
    if fno_label: print(f"  FNO: {fno_label}")
    
    for metric in metrics_of_interest:
        results[metric] = {}
        
        wno_values = all_runs[all_runs['Label'] == wno_label][metric].values
        
        if len(wno_values) < 2:
            continue
        
        if lstm_label:
            lstm_values = all_runs[all_runs['Label'] == lstm_label][metric].values
            try:
                t_stat, p_value = ttest_rel(wno_values, lstm_values)
                test_type = 'paired t-test'
            except:
                t_stat, p_value = ttest_ind(wno_values, lstm_values)
                test_type = 'independent t-test'
            
            pooled_std = np.sqrt((np.std(wno_values)**2 + np.std(lstm_values)**2) / 2)
            cohens_d = (np.mean(wno_values) - np.mean(lstm_values)) / (pooled_std + 1e-8)
            
            if 'Acc' in metric or 'Proximity' in metric:
                better = 'WNO' if np.mean(wno_values) > np.mean(lstm_values) else 'BiLSTM'
            else:
                better = 'WNO' if np.mean(wno_values) < np.mean(lstm_values) else 'BiLSTM'
            
            results[metric]['vs_BiLSTM'] = {
                'wno_mean': np.mean(wno_values),
                'wno_std': np.std(wno_values),
                'baseline_mean': np.mean(lstm_values),
                'baseline_std': np.std(lstm_values),
                'baseline_name': 'BiLSTM',
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'better': better,
                'test_type': test_type,
                'n_seeds': len(wno_values)
            }
        
        if trans_label:
            trans_values = all_runs[all_runs['Label'] == trans_label][metric].values
            try:
                t_stat, p_value = ttest_rel(wno_values, trans_values)
                test_type = 'paired t-test'
            except:
                t_stat, p_value = ttest_ind(wno_values, trans_values)
                test_type = 'independent t-test'
            
            pooled_std = np.sqrt((np.std(wno_values)**2 + np.std(trans_values)**2) / 2)
            cohens_d = (np.mean(wno_values) - np.mean(trans_values)) / (pooled_std + 1e-8)
            
            if 'Acc' in metric or 'Proximity' in metric:
                better = 'WNO' if np.mean(wno_values) > np.mean(trans_values) else 'Transformer'
            else:
                better = 'WNO' if np.mean(wno_values) < np.mean(trans_values) else 'Transformer'
            
            results[metric]['vs_Transformer'] = {
                'wno_mean': np.mean(wno_values),
                'wno_std': np.std(wno_values),
                'baseline_mean': np.mean(trans_values),
                'baseline_std': np.std(trans_values),
                'baseline_name': 'Transformer',
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'better': better,
                'test_type': test_type,
                'n_seeds': len(wno_values)
            }
        
        if fno_label:
            fno_values = all_runs[all_runs['Label'] == fno_label][metric].values
            try:
                t_stat, p_value = ttest_rel(wno_values, fno_values)
                test_type = 'paired t-test'
            except:
                t_stat, p_value = ttest_ind(wno_values, fno_values)
                test_type = 'independent t-test'
            
            pooled_std = np.sqrt((np.std(wno_values)**2 + np.std(fno_values)**2) / 2)
            cohens_d = (np.mean(wno_values) - np.mean(fno_values)) / (pooled_std + 1e-8)
            
            if 'Acc' in metric or 'Proximity' in metric:
                better = 'WNO' if np.mean(wno_values) > np.mean(fno_values) else 'FNO'
            else:
                better = 'WNO' if np.mean(wno_values) < np.mean(fno_values) else 'FNO'
            
            results[metric]['vs_FNO'] = {
                'wno_mean': np.mean(wno_values),
                'wno_std': np.std(wno_values),
                'baseline_mean': np.mean(fno_values),
                'baseline_std': np.std(fno_values),
                'baseline_name': 'FNO',
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'better': better,
                'test_type': test_type,
                'n_seeds': len(wno_values)
            }
    
    return results

def save_statistical_tests(results, metrics_of_interest, run_id):
    """Save statistical test results to CSV."""
    rows = []
    
    for metric in metrics_of_interest:
        if metric not in results or not results[metric]:
            continue
        
        for comparison, stats_dict in results[metric].items():
            rows.append({
                'Metric': metric,
                'Comparison': comparison,
                'WNO_Mean': stats_dict['wno_mean'],
                'WNO_Std': stats_dict['wno_std'],
                'Baseline': stats_dict['baseline_name'],
                'Baseline_Mean': stats_dict['baseline_mean'],
                'Baseline_Std': stats_dict['baseline_std'],
                't_statistic': stats_dict['t_statistic'],
                'p_value': stats_dict['p_value'],
                'Cohen_s_d': stats_dict['cohens_d'],
                'Significant': stats_dict['significant'],
                'Better': stats_dict['better'],
                'n_seeds': stats_dict['n_seeds'],
            })
    
    if rows:
        df = pd.DataFrame(rows)
        save_path = f"{LOGS_PATH}/{run_id}/analysis/statistical_tests.csv"
        df.to_csv(save_path, index=False)
        print(f"\nStatistical tests saved to: {save_path}")
        print(df.to_string())

def print_statistical_report(results, metrics_of_interest):
    """Print a formatted statistical report."""
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*80)
    
    metric_display = {
        'Ex_Loc_Mean': 'Spatial Localization Error',
        'Px_Loc_Mean': 'P(x) Localization Error',
        'Ex_Time_Mean': 'Temporal Onset Error Mean (hrs)',
        'Ex_Time_P90':  'Temporal Onset Error 90th Pct (hrs)',
        'Px_Top1_Acc': 'P(x) Top-1 Accuracy',
        'Px_Top3_Acc': 'P(x) Top-3 Accuracy',
        'Px_Top5_Acc': 'P(x) Top-5 Accuracy',
        # 'Px_Top10_Acc':'P(x) Top-10 Accuracy',
        'Px_Proximity_Acc': 'P(x) Proximity Accuracy',
        'Explainability_Fidelity_Delta': 'Explainability Fidelity Δ'
    }
    
    for metric in metrics_of_interest:
        if metric not in results or not results[metric]:
            continue
        
        print(f"\n{'─'*80}")
        print(f"METRIC: {metric_display.get(metric, metric)}")
        print(f"{'─'*80}")
        
        for comparison, stats_dict in results[metric].items():
            sig_symbol = "✓ SIGNIFICANT" if stats_dict.get('significant', False) else "✗ NOT SIGNIFICANT"
            baseline = stats_dict.get('baseline_name', 'Baseline')
            
            print(f"\n  {comparison}:")
            print(f"    WNO:        {stats_dict['wno_mean']:.4f} ± {stats_dict['wno_std']:.4f}")
            print(f"    {baseline}: {stats_dict['baseline_mean']:.4f} ± {stats_dict['baseline_std']:.4f}")
            print(f"    t-statistic: {stats_dict['t_statistic']:.4f}")
            print(f"    p-value:     {stats_dict['p_value']:.4f} ({sig_symbol})")
            print(f"    Cohen's d:   {stats_dict['cohens_d']:.4f}")
            print(f"    Better:      {stats_dict['better']}")

def plot_results(all_runs, metrics_of_interest, run_id):
    """Create visualization plots."""
    os.makedirs(f"{LOGS_PATH}/{run_id}/analysis", exist_ok=True)
    
    # Define which metrics are for P(x) only
    px_metrics = ['Px_Top1_Acc', 'Px_Top3_Acc', 'Px_Top5_Acc', 'Px_Proximity_Acc']
    
    plot_configs = [
        {'metric': 'Ex_Loc_Mean', 'title': 'Spatial Localization Error', 'filter_baselines': False},
        {'metric': 'Px_Top5_Acc', 'title': 'P(x) Top-5 Accuracy', 'filter_baselines': True},
        {'metric': 'Px_Proximity_Acc', 'title': 'P(x) Proximity Accuracy', 'filter_baselines': True},
    ]
    
    for config in plot_configs:
        metric = config['metric']
        title = config['title']
        filter_baselines = config['filter_baselines']
        
        if metric not in all_runs.columns:
            continue
        
        plt.figure(figsize=(10, 6))
        
        labels = all_runs['Label'].unique()
        data = []
        label_names = []
        
        for label in labels:
            if filter_baselines:
                if any(b in label for b in ['BiLSTM', 'Transformer', 'FNO', 'IDW']):
                    continue
            
            values = all_runs[all_runs['Label'] == label][metric].values
            if len(values) > 0:
                data.append(values)
                clean_label = label.replace('_Validation', '').replace('_Unseen_Test', '')
                if len(clean_label) > 25:
                    clean_label = clean_label[:25]
                label_names.append(clean_label)
        
        if not data:
            print(f"⚠️ No data for {metric}")
            continue
        
        bp = plt.boxplot(data, patch_artist=True, labels=label_names)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23']
        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        plt.ylabel(title)
        plt.xlabel('Model / Dataset')
        plt.title(f'{title} Across {len(all_runs["seed"].unique())} Seeds')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f"{LOGS_PATH}/{run_id}/analysis/{metric}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

def main():
    """Main function."""
    
    print("\n" + "="*80)
    print("MULTI-SEED RESULTS ANALYZER")
    print("="*80)
    
    run_list = list_all_runs()
    
    if not run_list:
        print("\nNo runs found!")
        print(f"Current logs path: {LOGS_PATH}")
        return
    
    run_id = select_run_interactive(run_list)
    
    if not run_id:
        print("No run selected. Exiting.")
        return
    
    print(f"\nAnalyzing RUN_ID: {run_id}")
    
    metrics_of_interest = [
    "Ex_Loc_Mean",
    "Ex_Time_Mean",
    "Ex_Time_P90",
    "Px_Loc_Mean",
    "Px_Top1_Acc",
    "Px_Top3_Acc",
    "Px_Top5_Acc",
    # "Px_Top10_Acc",
    "Px_Proximity_Acc",
    "Explainability_Fidelity_Delta",
    ]
    
    all_runs = load_all_seeds(run_id)
    
    if all_runs is None or all_runs.empty:
        print("No data found. Exiting.")
        return
    
    print(f"\nLoaded {len(all_runs)} rows from {len(all_runs['seed'].unique())} seeds")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    summary = create_summary_table(all_runs, metrics_of_interest, run_id)
    print(summary)
    
    print("\n" + "="*80)
    print("COMPUTING STATISTICAL TESTS")
    print("="*80)
    results = compute_statistical_tests(all_runs, metrics_of_interest)
    
    print_statistical_report(results, metrics_of_interest)
    
    # Save statistical tests to CSV
    save_statistical_tests(results, metrics_of_interest, run_id)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    plot_results(all_runs, metrics_of_interest, run_id)
    
    all_runs.to_csv(f"{LOGS_PATH}/{run_id}/analysis/all_seed_data.csv", index=False)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {LOGS_PATH}/{run_id}/analysis/")
    print("="*80)

if __name__ == "__main__":
    main()