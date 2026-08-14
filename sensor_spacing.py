# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:39:53 2026

@author: anjum
"""

# run_sensor_spacing.py
import numpy as np
from real_dataset_loader import scan_and_process_folder

def compute_sensor_spacing(coords):
    N = len(coords)
    R = 6371.0
    distances = np.full((N, N), np.inf)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            lat1, lon1 = np.deg2rad(coords[i])
            lat2, lon2 = np.deg2rad(coords[j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (np.sin(dlat/2)**2 + 
                 np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2)
            distances[i,j] = 2 * R * np.arcsin(np.sqrt(a.clip(0,1)))
    
    nn_dist = distances.min(axis=1)
    return nn_dist.mean(), nn_dist.std(), nn_dist.min(), nn_dist.max()

# Load from cache -- no reprocessing
all_regions = scan_and_process_folder(
    data_folder="G:/My Drive/WNO Project/Datafolder",
    force_reprocess=False  # uses cache
)

print(f"\n{'City':15s} {'Year':6s} {'N':6s} {'Mean spacing':15s} {'Std':10s} {'Min':10s} {'Max':10s}")
print("-" * 75)

for r in all_regions:
    mean_s, std_s, min_s, max_s = compute_sensor_spacing(r['coords'])
    print(f"{r['name']:15s} {r['year']:6d} {r['coords'].shape[0]:6d} "
          f"{mean_s:10.2f} km    {std_s:8.2f}   {min_s:8.2f}   {max_s:8.2f}")
    
import pandas as pd
import os

results = []
for r in all_regions:
    mean_s, std_s, min_s, max_s = compute_sensor_spacing(r['coords'])
    results.append({
        'City':         r['name'],
        'Region':       r['region'],
        'Year':         r['year'],
        'N_Sensors':    r['coords'].shape[0],
        'N_Events':     r['U'].shape[0],
        'Mean_Spacing_km': round(mean_s, 3),
        'Std_Spacing_km':  round(std_s, 3),
        'Min_Spacing_km':  round(min_s, 3),
        'Max_Spacing_km':  round(max_s, 3),
    })

df = pd.DataFrame(results)

# Add theoretical lower bound column
# Lower bound = mean nearest-neighbor spacing
df['Theoretical_LB_km'] = df['Mean_Spacing_km']

# Save
save_path = "G:/My Drive/WNO Project/sensor_spacing_analysis.csv"
df.to_csv(save_path, index=False)
print(f"\nSaved to: {save_path}")
print(df.to_string(index=False))