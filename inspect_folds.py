# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:36:54 2026

@author: anjum
"""

# inspect_folds.py
import config
from real_dataset_loader import scan_and_process_folder, build_fold

all_regions = scan_and_process_folder(
    data_folder=config.DATAFOLDER,
    window_length=config.T,
    force_reprocess=False,
)

print(f"\nTotal events available: {sum(r['U'].shape[0] for r in all_regions)}")
print(f"Total region-years: {len(all_regions)}")

for fold in ['US_TO_EUROPE', 'EUROPE_TO_US', 'WITHIN_EU', 'WITHIN_US', 'TEMPORAL']:
    train, val, test, labels = build_fold(all_regions, fold=fold)
    train_n = sum(r['U'].shape[0] for r in train)
    val_n   = sum(r['U'].shape[0] for r in val)
    test_n  = sum(r['U'].shape[0] for r in test)
    print(f"\n{fold}:")
    print(f"  Train: {train_n} events from {len(train)} region-years")
    print(f"  Val:   {val_n} events from {len(val)} region-years")
    print(f"  Test:  {test_n} events from {len(test)} region-years")