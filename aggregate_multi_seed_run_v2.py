# -*- coding: utf-8 -*-
"""
Multi-Seed Results Aggregator with Statistical Tests.
Updated: val/test separation, IDW/Peak baselines, adversarial
tests, wavelet weights summary. NO statistical tests for P(x)
metrics since baselines produce no P(x) measurements.
@author: anjum
"""
import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import ttest_rel, ttest_ind
import re
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
_DEFAULTS = {'LOGS_PATH': "logs"}

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOGS_PATH', type=str,
                        default=_DEFAULTS['LOGS_PATH'])
    args, _ = parser.parse_known_args()
    return args

_args = _parse_args()
LOGS_PATH = _args.LOGS_PATH

# ============================================================================
# METRICS CONFIGURATION
# — Statistical tests run ONLY on these metrics
# — P(x) metrics reported in summary but NOT tested
# ============================================================================
STAT_TEST_METRICS = [
    "Ex_Loc_Mean",
    "Ex_Time_Mean",
    "Ex_Time_P90",
    "Explainability_Fidelity_Delta",
]

REPORT_ONLY_METRICS = [
    "Px_Top1_Acc",
    "Px_Top3_Acc",
    "Px_Top5_Acc",
    "Px_Proximity_Acc",
    "Px_Loc_Mean",
]

ALL_METRICS = STAT_TEST_METRICS + REPORT_ONLY_METRICS

# ============================================================================
# LABEL CLASSIFICATION
# ============================================================================
TEST_LABEL_PATTERNS = [
    'Unseen_Test',
    'unseen_test',
    '2024_Unseen',
]

ADVERSARIAL_PATTERNS = [
    'Adversarial',
    'adversarial',
    'Decoy',
]

def is_test_label(label):
    return any(p in label for p in TEST_LABEL_PATTERNS)

def is_adversarial_label(label):
    return any(p in label for p in ADVERSARIAL_PATTERNS)

def is_val_label(label):
    return not is_test_label(label) and not is_adversarial_label(label)

def get_model_type(label):
    if 'BiLSTM' in label:
        return 'BiLSTM'
    elif 'Transformer' in label:
        return 'Transformer'
    elif 'FNO' in label:
        return 'FNO'
    elif 'IDW' in label:
        return 'IDW'
    elif 'Peak' in label:
        return 'Peak'
    else:
        return 'WNO'

def get_wno_label(labels, split='val'):
    if split == 'val':
        wno = [l for l in labels
               if get_model_type(l) == 'WNO' and is_val_label(l)]
    elif split == 'test':
        wno = [l for l in labels
               if get_model_type(l) == 'WNO' and is_test_label(l)]
    elif split == 'adversarial':
        wno = [l for l in labels
               if get_model_type(l) == 'WNO' and is_adversarial_label(l)]
    else:
        wno = []
    return wno[0] if wno else None

def get_baseline_label(labels, model_type, split='val'):
    candidates = [l for l in labels if get_model_type(l) == model_type]
    if split == 'val':
        candidates = [l for l in candidates if is_val_label(l)]
    elif split == 'test':
        candidates = [l for l in candidates if is_test_label(l)]
    elif split == 'adversarial':
        candidates = [l for l in candidates if is_adversarial_label(l)]
    return candidates[0] if candidates else None

# ============================================================================
# FILE LOADING
# ============================================================================
def list_all_runs():
    if not os.path.exists(LOGS_PATH):
        print(f"Logs directory not found: {LOGS_PATH}")
        return []
    log_dirs = glob.glob(f"{LOGS_PATH}/*")
    run_list = []
    for d in log_dirs:
        run_id = os.path.basename(d)
        seeds = glob.glob(f"{d}/seed*")
        has_results = any(
            os.path.exists(os.path.join(s, "evaluation_metrics.csv"))
            for s in seeds
        )
        try:
            dt = datetime.strptime(run_id, '%Y-%m-%d_%H%M%S')
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = "Unknown"
        run_list.append({
            'run_id': run_id,
            'num_seeds': len(seeds),
            'has_results': has_results,
            'time': time_str,
            'path': d
        })
    return run_list

def display_runs_table(run_list):
    print("\n" + "="*80)
    print("AVAILABLE RUNS")
    print("="*80)
    print(f"{'#':<3} {'RUN_ID':<25} {'Seeds':<6} {'Results':<8} {'Time':<20}")
    print("-"*80)
    for i, run in enumerate(run_list):
        status = "✅" if run['has_results'] else "❌"
        print(f"{i:<3} {run['run_id']:<25} {run['num_seeds']:<6} "
              f"{status:<8} {run['time']:<20}")
    print("="*80)

def select_run_interactive(run_list):
    valid_runs = [r for r in run_list if r['has_results']]
    if not valid_runs:
        print("No runs with complete results found!")
        return None
    display_runs_table(valid_runs)
    if len(valid_runs) == 1:
        print(f"\nOnly one valid run. Using: {valid_runs[0]['run_id']}")
        return valid_runs[0]['run_id']
    while True:
        try:
            choice = input(
                "\nSelect RUN_ID (number or 'latest'): "
            ).strip()
            if choice.lower() == 'latest':
                valid_runs.sort(key=lambda x: x['time'], reverse=True)
                return valid_runs[0]['run_id']
            idx = int(choice)
            if 0 <= idx < len(valid_runs):
                return valid_runs[idx]['run_id']
            print(f"Enter 0-{len(valid_runs)-1}")
        except ValueError:
            print("Invalid input.")

def load_all_seeds(run_id):
    run_path = os.path.join(LOGS_PATH, run_id)
    
    corrected_files = glob.glob(
        f"{run_path}/seed*/evaluation_metrics_corrected.csv"
    )
    original_files = glob.glob(
        f"{run_path}/seed*/evaluation_metrics.csv"
    )
    
    if not original_files:
        print(f"Warning: No files found for {run_id}")
        return None
    
    print(f"\nFound {len(original_files)} original + "
          f"{len(corrected_files)} corrected seed files")
    
    dfs = []
    for orig_f in original_files:
        seed_match = re.search(r'seed(\d+)', orig_f)
        seed = int(seed_match.group(1)) if seed_match else -1
        
        try:
            orig_df = pd.read_csv(orig_f, engine='python')
            orig_df['seed'] = seed
            
            # Find matching corrected file for this seed
            corr_f = orig_f.replace(
                'evaluation_metrics.csv',
                'evaluation_metrics_corrected.csv'
            )
            
            if os.path.exists(corr_f):
                corr_df = pd.read_csv(corr_f, engine='python')
                corr_df['seed'] = seed
                
                # Get WNO labels from corrected file
                wno_labels = corr_df['Label'].unique().tolist()
                
                # Remove old WNO rows from original
                label_col = next(
                    (c for c in orig_df.columns
                     if c.lower() == 'label'), 'Label'
                )
                orig_df = orig_df[
                    ~orig_df[label_col].isin(wno_labels)
                ]
                
                # Combine baseline rows + corrected WNO rows
                merged = pd.concat(
                    [orig_df, corr_df], ignore_index=True
                )
                dfs.append(merged)
                print(f"  Seed {seed}: merged "
                      f"{len(orig_df)} baseline + "
                      f"{len(corr_df)} corrected WNO rows")
            else:
                # No corrected file — use original as-is
                dfs.append(orig_df)
                print(f"  Seed {seed}: using original only")
                
        except Exception as e:
            print(f"Warning: Could not read {orig_f}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else None

# ============================================================================
# WAVELET WEIGHTS SUMMARY
# ============================================================================
def extract_wavelet_weights(all_runs):
    """
    Extract and summarize wavelet band weights across seeds.
    Weights are stored as string '[0.0, w1, w2, w3]' in
    Event_N_Wavelet_Weights column.
    """
    if 'Event_N_Wavelet_Weights' not in all_runs.columns:
        print("No wavelet weights column found.")
        return None

    records = []
    wno_rows = all_runs[all_runs['Label'].apply(
        lambda l: get_model_type(l) == 'WNO'
    )]

    for _, row in wno_rows.iterrows():
        raw = str(row['Event_N_Wavelet_Weights'])
        try:
            weights = [float(x.strip()) for x in
                       raw.strip('[]').split(',')]
            if len(weights) >= 4:
                records.append({
                    'Label': row['Label'],
                    'seed':  row.get('seed', -1),
                    'Split': ('test' if is_test_label(row['Label'])
                              else 'adversarial'
                              if is_adversarial_label(row['Label'])
                              else 'val'),
                    'w0': weights[0],
                    'w1': weights[1],
                    'w2': weights[2],
                    'w3': weights[3],
                })
        except:
            continue

    if not records:
        return None

    df = pd.DataFrame(records)
    summary = df.groupby(['Label', 'Split'])[
        ['w0', 'w1', 'w2', 'w3']
    ].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]

    # Flag uniform weights (domain mismatch diagnostic)
    for band in ['w1', 'w2', 'w3']:
        summary[f'{band}_uniform'] = (
            summary[f'{band}_mean'].between(0.30, 0.36)
        )
    summary['is_uniform'] = (
        summary['w1_uniform'] &
        summary['w2_uniform'] &
        summary['w3_uniform']
    )

    return summary

def print_wavelet_summary(wavelet_summary):
    if wavelet_summary is None:
        return
    print(f"\n{'='*80}")
    print("WAVELET BAND WEIGHT SUMMARY")
    print("(Uniform ≈ [0, 0.33, 0.33, 0.33] signals domain mismatch)")
    print(f"{'='*80}")
    print(f"{'Label':<45} {'Split':<6} "
          f"{'w1':>8} {'w2':>8} {'w3':>8} {'Uniform?':>10}")
    print("-"*90)
    for (label, split), row in wavelet_summary.iterrows():
        uniform = "⚠️ YES" if row['is_uniform'] else "✅ NO"
        print(f"{label:<45} {split:<6} "
              f"{row['w1_mean']:>8.3f} "
              f"{row['w2_mean']:>8.3f} "
              f"{row['w3_mean']:>8.3f} "
              f"{uniform:>10}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
def create_summary_table(all_runs, run_id):
    """
    Full summary: val, test, adversarial splits.
    Includes ALL metrics including P(x) — reported not tested.
    Also includes E(t) and wavelet weights.
    """
    os.makedirs(f"{LOGS_PATH}/{run_id}/analysis_v2", exist_ok=True)

    all_runs = all_runs.copy()
    all_runs['Split'] = all_runs['Label'].apply(
        lambda l: ('test' if is_test_label(l)
                   else 'adversarial' if is_adversarial_label(l)
                   else 'val')
    )
    all_runs['Model_Type'] = all_runs['Label'].apply(get_model_type)

    summary = all_runs.groupby(
        ['Label', 'Split', 'Model_Type']
    )[ALL_METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary.to_csv(
        f"{LOGS_PATH}/{run_id}/analysis_v2/multiseed_summary.csv"
    )
    print(f"\nSummary saved to: "
          f"{LOGS_PATH}/{run_id}/analysis_v2/multiseed_summary.csv")
    return summary, all_runs

# ============================================================================
# STATISTICAL TESTS
# — ONLY on STAT_TEST_METRICS
# — P(x) metrics are NEVER tested
# ============================================================================
def _run_ttest(wno_values, baseline_values, metric, baseline_name):
    try:
        t_stat, p_value = ttest_rel(wno_values, baseline_values)
        test_type = 'paired t-test'
    except:
        t_stat, p_value = ttest_ind(wno_values, baseline_values)
        test_type = 'independent t-test'

    pooled_std = np.sqrt(
        (np.std(wno_values)**2 + np.std(baseline_values)**2) / 2
    )
    cohens_d = (
        (np.mean(wno_values) - np.mean(baseline_values))
        / (pooled_std + 1e-8)
    )
    better = ('WNO' if np.mean(wno_values) < np.mean(baseline_values)
              else baseline_name)

    return {
        'wno_mean':      np.mean(wno_values),
        'wno_std':       np.std(wno_values),
        'baseline_mean': np.mean(baseline_values),
        'baseline_std':  np.std(baseline_values),
        'baseline_name': baseline_name,
        't_statistic':   t_stat,
        'p_value':       p_value,
        'cohens_d':      cohens_d,
        'significant':   p_value < 0.05,
        'better':        better,
        'test_type':     test_type,
        'n_seeds':       len(wno_values),
    }

def compute_statistical_tests(all_runs, run_id):
    """
    Statistical tests on STAT_TEST_METRICS only.
    Covers val, test, and adversarial splits.
    IDW and Peak tested on Ex_Loc_Mean and Ex_Time_Mean only.
    P(x) metrics are NEVER tested here.
    """
    labels = all_runs['Label'].unique()
    results = {'val': {}, 'test': {}, 'adversarial': {}}

    for split in ['val', 'test', 'adversarial']:
        if split == 'val':
            split_runs = all_runs[
                all_runs['Label'].apply(is_val_label)
            ]
        elif split == 'test':
            split_runs = all_runs[
                all_runs['Label'].apply(is_test_label)
            ]
        else:
            split_runs = all_runs[
                all_runs['Label'].apply(is_adversarial_label)
            ]

        split_labels = split_runs['Label'].unique()
        wno_label = get_wno_label(split_labels, split)

        if not wno_label:
            if split != 'adversarial':
                print(f"Warning: No WNO {split} label found.")
            continue

        print(f"\n  WNO {split}: {wno_label}")

        for metric in STAT_TEST_METRICS:  # P(x) NEVER here
            results[split][metric] = {}
            wno_values = split_runs[
                split_runs['Label'] == wno_label
            ][metric].values

            if len(wno_values) < 2:
                continue

            # BiLSTM, FNO, Transformer
            for model_name in ['BiLSTM', 'FNO', 'Transformer']:
                bl_label = get_baseline_label(
                    split_labels, model_name, split
                )
                if not bl_label:
                    continue
                bl_values = split_runs[
                    split_runs['Label'] == bl_label
                ][metric].values
                if len(bl_values) < 2:
                    continue
                results[split][metric][f'vs_{model_name}'] = \
                    _run_ttest(wno_values, bl_values,
                               metric, model_name)

            # IDW — Ex_Loc_Mean and Ex_Time_Mean only
            if metric in ['Ex_Loc_Mean', 'Ex_Time_Mean']:
                idw_label = get_baseline_label(
                    split_labels, 'IDW', split
                )
                if idw_label:
                    idw_values = split_runs[
                        split_runs['Label'] == idw_label
                    ][metric].values
                    if len(idw_values) >= 2:
                        results[split][metric]['vs_IDW'] = \
                            _run_ttest(wno_values, idw_values,
                                       metric, 'IDW')

            # Peak — Ex_Loc_Mean only, from Peak_Loc_Mean column
            if metric == 'Ex_Loc_Mean' and \
                    'Peak_Loc_Mean' in split_runs.columns:
                peak_values = split_runs[
                    split_runs['Label'] == wno_label
                ]['Peak_Loc_Mean'].values
                peak_values = peak_values[peak_values > 0]
                if len(peak_values) >= 2:
                    results[split][metric]['vs_Peak'] = \
                        _run_ttest(wno_values, peak_values,
                                   metric, 'Peak')

    return results

# ============================================================================
# ADVERSARIAL ROBUSTNESS TESTS
# Compare each model normal vs adversarial condition
# ============================================================================
def compute_adversarial_robustness(all_runs):
    """
    For synthetic data: compare normal vs adversarial for each model.
    Tests only Ex_Loc_Mean and Ex_Time_Mean.
    """
    labels = all_runs['Label'].unique()
    has_adversarial = any(is_adversarial_label(l) for l in labels)
    if not has_adversarial:
        return {}

    results = {}
    models = ['WNO', 'BiLSTM', 'FNO', 'Transformer', 'IDW']

    for model in models:
        normal_label = get_baseline_label(labels, model, 'val') \
            if model != 'WNO' else get_wno_label(labels, 'val')
        adv_label = get_baseline_label(labels, model, 'adversarial') \
            if model != 'WNO' else get_wno_label(labels, 'adversarial')

        if not normal_label or not adv_label:
            continue

        results[model] = {}
        for metric in ['Ex_Loc_Mean', 'Ex_Time_Mean']:
            normal_vals = all_runs[
                all_runs['Label'] == normal_label
            ][metric].values
            adv_vals = all_runs[
                all_runs['Label'] == adv_label
            ][metric].values

            if len(normal_vals) < 2 or len(adv_vals) < 2:
                continue

            try:
                t_stat, p_value = ttest_rel(normal_vals, adv_vals)
            except:
                t_stat, p_value = ttest_ind(normal_vals, adv_vals)

            pooled_std = np.sqrt(
                (np.std(normal_vals)**2 + np.std(adv_vals)**2) / 2
            )
            cohens_d = (
                (np.mean(normal_vals) - np.mean(adv_vals))
                / (pooled_std + 1e-8)
            )

            results[model][metric] = {
                'normal_mean': np.mean(normal_vals),
                'normal_std':  np.std(normal_vals),
                'adv_mean':    np.mean(adv_vals),
                'adv_std':     np.std(adv_vals),
                't_statistic': t_stat,
                'p_value':     p_value,
                'cohens_d':    cohens_d,
                'significant': p_value < 0.05,
                'degraded':    np.mean(adv_vals) > np.mean(normal_vals),
                'n_seeds':     len(normal_vals),
            }

        # P(x) for WNO adversarial — report only, no test vs baseline
        if model == 'WNO':
            for px_metric in ['Px_Top1_Acc', 'Px_Top5_Acc',
                              'Px_Proximity_Acc']:
                normal_vals = all_runs[
                    all_runs['Label'] == normal_label
                ][px_metric].values
                adv_vals = all_runs[
                    all_runs['Label'] == adv_label
                ][px_metric].values
                if len(normal_vals) < 1 or len(adv_vals) < 1:
                    continue
                results[model][px_metric] = {
                    'normal_mean': np.mean(normal_vals),
                    'normal_std':  np.std(normal_vals),
                    'adv_mean':    np.mean(adv_vals),
                    'adv_std':     np.std(adv_vals),
                    'note': 'reported only — no baseline comparison',
                }

    return results

def print_adversarial_report(results):
    if not results:
        return
    print(f"\n{'='*80}")
    print("ADVERSARIAL ROBUSTNESS — Normal vs Adversarial")
    print(f"{'='*80}")
    for model, metrics in results.items():
        print(f"\n  Model: {model}")
        print(f"  {'Metric':<35} {'Normal':>12} {'Adv':>12} "
              f"{'p-value':>10} {'Significant':>12} {'Degraded?':>10}")
        print(f"  {'-'*85}")
        for metric, sd in metrics.items():
            if 'p_value' in sd:
                sig = "✓" if sd['significant'] else "✗"
                deg = "⚠️ YES" if sd['degraded'] else "✅ NO"
                print(f"  {metric:<35} "
                      f"{sd['normal_mean']:>12.4f} "
                      f"{sd['adv_mean']:>12.4f} "
                      f"{sd['p_value']:>10.4f} "
                      f"{sig:>12} "
                      f"{deg:>10}")
            else:
                print(f"  {metric:<35} "
                      f"{sd['normal_mean']:>12.2f}% "
                      f"{sd['adv_mean']:>12.2f}% "
                      f"{'(report only)':>22}")

def save_adversarial_tests(results, run_id):
    rows = []
    for model, metrics in results.items():
        for metric, sd in metrics.items():
            row = {
                'Model':        model,
                'Metric':       metric,
                'Normal_Mean':  sd['normal_mean'],
                'Normal_Std':   sd['normal_std'],
                'Adv_Mean':     sd['adv_mean'],
                'Adv_Std':      sd['adv_std'],
            }
            if 'p_value' in sd:
                row.update({
                    't_statistic': sd['t_statistic'],
                    'p_value':     sd['p_value'],
                    'Cohen_s_d':   sd['cohens_d'],
                    'Significant': sd['significant'],
                    'Degraded':    sd['degraded'],
                })
            else:
                row['Note'] = sd.get('note', '')
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        path = f"{LOGS_PATH}/{run_id}/analysis_v2/adversarial_tests.csv"
        df.to_csv(path, index=False)
        print(f"\nAdversarial tests saved to: {path}")

# ============================================================================
# REPORTING
# ============================================================================
def save_statistical_tests(results, run_id):
    rows = []
    for split in ['val', 'test', 'adversarial']:
        for metric in STAT_TEST_METRICS:
            if metric not in results.get(split, {}):
                continue
            for comparison, sd in results[split][metric].items():
                if 'p_value' not in sd:
                    continue
                rows.append({
                    'Split':         split,
                    'Metric':        metric,
                    'Comparison':    comparison,
                    'WNO_Mean':      sd['wno_mean'],
                    'WNO_Std':       sd['wno_std'],
                    'Baseline':      sd['baseline_name'],
                    'Baseline_Mean': sd['baseline_mean'],
                    'Baseline_Std':  sd['baseline_std'],
                    't_statistic':   sd['t_statistic'],
                    'p_value':       sd['p_value'],
                    'Cohen_s_d':     sd['cohens_d'],
                    'Significant':   sd['significant'],
                    'Better':        sd['better'],
                    'n_seeds':       sd['n_seeds'],
                })
    if rows:
        df = pd.DataFrame(rows)
        path = (f"{LOGS_PATH}/{run_id}/analysis_v2/"
                f"statistical_tests.csv")
        df.to_csv(path, index=False)
        print(f"\nStatistical tests saved to: {path}")
        print(df.to_string())

def print_statistical_report(results):
    metric_display = {
        'Ex_Loc_Mean':  'E(x) Spatial Localization Error',
        'Ex_Time_Mean': 'E(t) Temporal Onset Error Mean (hrs)',
        'Ex_Time_P90':  'E(t) Temporal Onset Error P90 (hrs)',
        'Explainability_Fidelity_Delta': 'Explainability Fidelity Δ',
    }
    for split in ['val', 'test', 'adversarial']:
        if not results.get(split):
            continue
        print(f"\n{'='*80}")
        print(f"STATISTICAL REPORT — {split.upper()} SET")
        print("NOTE: P(x) metrics not tested — baselines produce no P(x)")
        print(f"{'='*80}")
        for metric in STAT_TEST_METRICS:
            if metric not in results[split] or \
                    not results[split][metric]:
                continue
            print(f"\n{'─'*80}")
            print(f"METRIC: {metric_display.get(metric, metric)}")
            print(f"{'─'*80}")
            for comparison, sd in results[split][metric].items():
                if 'p_value' not in sd:
                    continue
                sig = ("✓ SIGNIFICANT"
                       if sd['significant'] else "✗ NOT SIGNIFICANT")
                print(f"\n  {comparison}:")
                print(f"    WNO:         {sd['wno_mean']:.4f}"
                      f" ± {sd['wno_std']:.4f}")
                print(f"    {sd['baseline_name']:<12}: "
                      f"{sd['baseline_mean']:.4f}"
                      f" ± {sd['baseline_std']:.4f}")
                print(f"    t-statistic: {sd['t_statistic']:.4f}")
                print(f"    p-value:     {sd['p_value']:.4f} ({sig})")
                print(f"    Cohen's d:   {sd['cohens_d']:.4f}")
                print(f"    Better:      {sd['better']}")

def print_paper_ready_table(all_runs):
    print(f"\n{'='*80}")
    print("PAPER-READY SUMMARY")
    print("P(x) reported for WNO only — no baseline comparison")
    print(f"{'='*80}")

    header = (f"{'Model':<20} {'Split':<12} "
              f"{'E(x)':>10} {'±':>4} "
              f"{'E(t)':>8} {'±':>4} "
              f"{'Top-1':>7} {'Top-5':>7} {'Prox':>7}")
    print(header)
    print("-"*85)

    model_order = ['IDW', 'Peak', 'BiLSTM', 'FNO', 'Transformer', 'WNO']

    for split in ['val', 'test', 'adversarial']:
        if split == 'val':
            split_runs = all_runs[
                all_runs['Label'].apply(is_val_label)
            ]
        elif split == 'test':
            split_runs = all_runs[
                all_runs['Label'].apply(is_test_label)
            ]
        else:
            split_runs = all_runs[
                all_runs['Label'].apply(is_adversarial_label)
            ]

        if split_runs.empty:
            continue

        split_labels = split_runs['Label'].unique()

        for model in model_order:
            if model == 'Peak':
                wno_lbl = get_wno_label(split_labels, split)
                if wno_lbl and 'Peak_Loc_Mean' in split_runs.columns:
                    pv = split_runs[
                        split_runs['Label'] == wno_lbl
                    ]['Peak_Loc_Mean'].values
                    pv = pv[pv > 0]
                    if len(pv) > 0:
                        print(f"{'Peak Sensor':<20} {split:<12} "
                              f"{np.mean(pv):>10.3f} "
                              f"{np.std(pv):>4.3f} "
                              f"{'---':>8} {'':>4} "
                              f"{'---':>7} {'---':>7} {'---':>7}")
                continue

            lbl = (get_wno_label(split_labels, split)
                   if model == 'WNO'
                   else get_baseline_label(split_labels, model, split))
            if not lbl:
                continue

            rows = split_runs[split_runs['Label'] == lbl]
            ex_m = rows['Ex_Loc_Mean'].mean()
            ex_s = rows['Ex_Loc_Mean'].std()
            et_m = rows['Ex_Time_Mean'].mean()
            et_s = rows['Ex_Time_Mean'].std()

            if model == 'WNO':
                t1 = f"{rows['Px_Top1_Acc'].mean():.1f}%"
                t5 = f"{rows['Px_Top5_Acc'].mean():.1f}%"
                pr = f"{rows['Px_Proximity_Acc'].mean():.1f}%"
            elif model == 'IDW':
                t1 = t5 = pr = '---'
                et_m = float('nan')
                et_s = float('nan')
            else:
                t1 = t5 = pr = '---'

            et_str = (f"{et_m:>8.3f} {et_s:>4.3f}"
                      if not np.isnan(et_m) else f"{'---':>8} {'':>4}")
            print(f"{model:<20} {split:<12} "
                  f"{ex_m:>10.3f} {ex_s:>4.3f} "
                  f"{et_str} "
                  f"{t1:>7} {t5:>7} {pr:>7}")

        print()

# ============================================================================
# PLOTS
# ============================================================================
def plot_results(all_runs, run_id):
    os.makedirs(f"{LOGS_PATH}/{run_id}/analysis_v2", exist_ok=True)

    plot_configs = [
        {'metric': 'Ex_Loc_Mean',
         'title': 'E(x) Spatial Localization Error',
         'wno_only': False},
        {'metric': 'Ex_Time_Mean',
         'title': 'E(t) Temporal Onset Error',
         'wno_only': False},
        {'metric': 'Px_Top5_Acc',
         'title': 'P(x) Top-5 Accuracy (WNO only)',
         'wno_only': True},
        {'metric': 'Px_Proximity_Acc',
         'title': 'P(x) Proximity Accuracy (WNO only)',
         'wno_only': True},
    ]

    for cfg in plot_configs:
        metric = cfg['metric']
        if metric not in all_runs.columns:
            continue

        plt.figure(figsize=(12, 6))
        data, label_names = [], []

        for label in all_runs['Label'].unique():
            if cfg['wno_only'] and get_model_type(label) != 'WNO':
                continue
            values = all_runs[
                all_runs['Label'] == label
            ][metric].values
            if len(values) > 0:
                data.append(values)
                split = ('test' if is_test_label(label)
                         else 'adv' if is_adversarial_label(label)
                         else 'val')
                label_names.append(
                    f"{get_model_type(label)}\n({split})"
                )

        if not data:
            continue

        bp = plt.boxplot(data, patch_artist=True, labels=label_names)
        colors = ['#2E86AB', '#A23B72', '#F18F01',
                  '#C73E1D', '#6B8E23', '#8B4513']
        for patch, color in zip(bp['boxes'], colors * 10):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        plt.ylabel(cfg['title'])
        plt.title(f"{cfg['title']} — Val / Test / Adversarial")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fname = (f"{LOGS_PATH}/{run_id}/analysis_v2/"
                 f"{metric}_summary.png")
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fname}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("MULTI-SEED RESULTS ANALYZER")
    print("Statistical tests: E(x), E(t), Fidelity only")
    print("P(x) metrics: reported in summary, NOT statistically tested")
    print("="*80)

    run_list = list_all_runs()
    if not run_list:
        print(f"\nNo runs found in {LOGS_PATH}")
        return

    run_id = select_run_interactive(run_list)
    if not run_id:
        return

    all_runs = load_all_seeds(run_id)
    if all_runs is None or all_runs.empty:
        print("No data found.")
        return

    print(f"\nLoaded {len(all_runs)} rows from "
          f"{len(all_runs['seed'].unique())} seeds")
    print(f"Labels found: {sorted(all_runs['Label'].unique())}")

    # 1 — Full summary table (all metrics, all splits)
    print("\n" + "="*80 + "\nSUMMARY TABLE\n" + "="*80)
    summary, all_runs = create_summary_table(all_runs, run_id)
    print(summary)

    # 2 — Wavelet weights
    print("\n" + "="*80 + "\nWAVELET WEIGHTS\n" + "="*80)
    wavelet_summary = extract_wavelet_weights(all_runs)
    print_wavelet_summary(wavelet_summary)
    if wavelet_summary is not None:
        wavelet_summary.to_csv(
            f"{LOGS_PATH}/{run_id}/analysis_v2/wavelet_weights.csv"
        )
        print(f"\nWavelet weights saved.")

    # 3 — Paper-ready table
    print_paper_ready_table(all_runs)

    # 4 — Statistical tests (E(x), E(t), Fidelity ONLY)
    print("\n" + "="*80)
    print("STATISTICAL TESTS — E(x), E(t), Fidelity ONLY")
    print("P(x) metrics deliberately excluded")
    print("="*80)
    results = compute_statistical_tests(all_runs, run_id)
    print_statistical_report(results)
    save_statistical_tests(results, run_id)

    # 5 — Adversarial robustness (synthetic only)
    has_adversarial = any(
        is_adversarial_label(l)
        for l in all_runs['Label'].unique()
    )
    if has_adversarial:
        print("\n" + "="*80)
        print("ADVERSARIAL ROBUSTNESS TESTS")
        print("="*80)
        adv_results = compute_adversarial_robustness(all_runs)
        print_adversarial_report(adv_results)
        save_adversarial_tests(adv_results, run_id)

    # 6 — Plots
    print("\n" + "="*80 + "\nCREATING VISUALIZATIONS\n" + "="*80)
    plot_results(all_runs, run_id)

    # 7 — Save full data
    all_runs.to_csv(
        f"{LOGS_PATH}/{run_id}/analysis_v2/all_seed_data.csv",
        index=False
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results in: {LOGS_PATH}/{run_id}/analysis_v2/")
    print("="*80)

if __name__ == "__main__":
    main()