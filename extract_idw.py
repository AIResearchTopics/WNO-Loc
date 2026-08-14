# -*- coding: utf-8 -*-
"""
IDW and Peak Sensor E(x) Extractor.
Reads both evaluation_metrics.csv and evaluation_report.txt files.
"""
import pandas as pd
import numpy as np
import glob
import os
import re

LOGS_PATH = 'logs'

def parse_report_txt(filepath):
    """
    Parse evaluation_report.txt and extract E(x) for all IDW profiles.
    Returns list of dicts with label and ex_loc_mean.
    """
    results = []
    current_label = None
    ex_loc_mean = None
    ex_loc_std = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect profile header
            match_label = re.search(
                r'PERFORMANCE REPORT PROFILE:\s*\[(.+?)\]', line
            )
            if match_label:
                # Save previous profile if it was IDW
                if current_label and 'IDW' in current_label and ex_loc_mean is not None:
                    results.append({
                        'label': current_label,
                        'ex_loc_mean': ex_loc_mean,
                        'ex_loc_std': ex_loc_std,
                    })
                current_label = match_label.group(1)
                ex_loc_mean = None
                ex_loc_std = None
                continue

            # Detect E(x) line
            match_ex = re.search(
                r'E\(x\) Mean Localization Error\s*:\s*([\d.]+)\s*\(std\s*([\d.]+)\)',
                line
            )
            if match_ex and current_label and 'IDW' in current_label:
                ex_loc_mean = float(match_ex.group(1))
                ex_loc_std = float(match_ex.group(2))

    # Save last profile
    if current_label and 'IDW' in current_label and ex_loc_mean is not None:
        results.append({
            'label': current_label,
            'ex_loc_mean': ex_loc_mean,
            'ex_loc_std': ex_loc_std,
        })

    return results


def extract_idw_peak(run_id):
    run_path = os.path.join(LOGS_PATH, run_id)

    # ------------------------------------------------------------------ #
    # Part 1 — CSV-based extraction (Peak Sensor from WNO rows)
    # ------------------------------------------------------------------ #
    csv_files = glob.glob(f"{run_path}/seed*/evaluation_metrics.csv")

    peak_normal     = []
    peak_adversarial = []
    idw_adv_csv     = []

    for f in csv_files:
        df = pd.read_csv(f)

        # Peak Sensor E(x) — normal condition (from Peak_Loc_Mean in WNO rows)
        wno_normal = df[df['Label'] == 'Validation_Normal']
        if not wno_normal.empty and 'Peak_Loc_Mean' in df.columns:
            peak_normal.append(wno_normal['Peak_Loc_Mean'].values[0])

        # Peak Sensor E(x) — adversarial condition
        wno_adv = df[df['Label'] == 'Validation_Adversarial_Decoy_1.5x']
        if not wno_adv.empty and 'Peak_Loc_Mean' in df.columns:
            peak_adversarial.append(wno_adv['Peak_Loc_Mean'].values[0])

        # IDW from CSV (adversarial only — legacy)
        idw_rows = df[df['Label'].str.contains('IDW', na=False)]
        for _, row in idw_rows.iterrows():
            idw_adv_csv.append({
                'label': row['Label'],
                'ex_loc_mean': row['Ex_Loc_Mean'],
            })

    # ------------------------------------------------------------------ #
    # Part 2 — TXT-based extraction (IDW normal and real-data folds)
    # ------------------------------------------------------------------ #
    txt_files = glob.glob(f"{run_path}/seed*/evaluation_report.txt")
    # Also check directly in seed folder root
    txt_files += glob.glob(f"{run_path}/seed*/*.txt")
    txt_files = list(set(txt_files))  # deduplicate

    idw_from_txt = {}  # label -> list of ex_loc_mean across seeds

    for f in txt_files:
        records = parse_report_txt(f)
        for rec in records:
            lbl = rec['label']
            if lbl not in idw_from_txt:
                idw_from_txt[lbl] = []
            idw_from_txt[lbl].append(rec['ex_loc_mean'])

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print(f"IDW AND PEAK SENSOR RESULTS — {run_id}")
    print("="*70)

    if peak_normal:
        print(f"\nPeak Sensor E(x) — Normal (synthetic):")
        print(f"  Per-seed: {[round(v,4) for v in peak_normal]}")
        print(f"  Mean:     {np.mean(peak_normal):.4f}")
        print(f"  Std:      {np.std(peak_normal):.4f}")

    if peak_adversarial:
        print(f"\nPeak Sensor E(x) — Adversarial (synthetic):")
        print(f"  Per-seed: {[round(v,4) for v in peak_adversarial]}")
        print(f"  Mean:     {np.mean(peak_adversarial):.4f}")
        print(f"  Std:      {np.std(peak_adversarial):.4f}")

    if idw_adv_csv:
        print(f"\nIDW E(x) — From CSV (adversarial synthetic only):")
        for rec in idw_adv_csv:
            print(f"  {rec['label']}: {rec['ex_loc_mean']:.4f}")

    if idw_from_txt:
        print(f"\nIDW E(x) — From TXT report files:")
        for lbl, values in sorted(idw_from_txt.items()):
            print(f"\n  Profile: {lbl}")
            print(f"  Per-seed: {[round(v,4) for v in values]}")
            print(f"  Mean:     {np.mean(values):.4f}")
            print(f"  Std:      {np.std(values):.4f}")
            print(f"  N seeds:  {len(values)}")
    else:
        print("\nNo IDW profiles found in TXT report files.")
        print("Check that evaluation_report.txt exists in seed folders.")

    print("\n" + "="*70)

    return {
        'peak_normal_mean':     np.mean(peak_normal) if peak_normal else None,
        'peak_normal_std':      np.std(peak_normal)  if peak_normal else None,
        'peak_adv_mean':        np.mean(peak_adversarial) if peak_adversarial else None,
        'idw_by_label':         {
            lbl: {
                'mean': np.mean(vals),
                'std':  np.std(vals),
                'n':    len(vals),
            }
            for lbl, vals in idw_from_txt.items()
        },
    }


if __name__ == "__main__":
    runs = [
        os.path.basename(d)
        for d in glob.glob(f"{LOGS_PATH}/*")
        if os.path.isdir(d)
    ]

    if not runs:
        print(f"No runs found in {LOGS_PATH}")
    elif len(runs) == 1:
        extract_idw_peak(runs[0])
    else:
        print("Available runs:")
        for i, r in enumerate(sorted(runs)):
            print(f"  {i}: {r}")
        idx = int(input("Select run number: "))
        extract_idw_peak(sorted(runs)[idx])