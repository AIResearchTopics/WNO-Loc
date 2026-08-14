# -*- coding: utf-8 -*-
"""
Advanced Evaluation Logger Module for Spatiotemporal Localization Frameworks.
Maintains synchronized text reports and comprehensive CSV spreadsheets for big data.
Target Venue: IEEE Big Data 2026.
@author: usman.anjum
"""

import os
import csv
import json

class EvaluationLogger:
    def __init__(self, output_dir):
        """
        Initializes disk directories and sets up fully named structural metric CSV tables.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.txt_path = os.path.join(self.output_dir, "evaluation_report.txt")
        self.csv_path = os.path.join(self.output_dir, "evaluation_metrics.csv")
        
        # Build the expanded header layout if creating the file fresh
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    # --- 1. Global Core Intensity Head Metrics E(x) ---
                    "Label", "Ex_Loc_Mean", "Ex_Loc_Std", "Ex_Loc_Median", "Ex_Loc_Normalized",
                    
                    # --- 2. Global Core Frequency Head Metrics P(x) [NEW COLUMNS] ---
                    "Px_Loc_Mean", "Px_Loc_Std", "Px_Loc_Median", "Px_Loc_Normalized",
                    
                    # --- 2b. Peak Sensor Baseline ---
                    "Peak_Loc_Mean", "Peak_Loc_Std", "Peak_Loc_Median",
                    
                    # --- 3. Global Temporal Onset Metrics ---
                    "Ex_Time_Mean", "Ex_Time_Std", "Ex_Time_P90",
                    
                    # --- 4. Attribution & Explainability Metrics ---
                    "Px_Top1_Acc", "Px_Top3_Acc", "Px_Top5_Acc","Px_Top10_Acc", "Px_Proximity_Acc", 
                    "Explainability_Fidelity_Delta", "Sensor_Dist_Mean", "Sensor_Dist_Std", "Sensor_Idx_Error",
                    
                    # --- 5. Global Explainability Profiles ---
                    "Freq_Mean", "Freq_Std", "Freq_Max", 
                    "Temp_Mean", "Temp_Std", "Temp_Max",
                    "Neigh_Mean", "Neigh_Std", "Neigh_Max",
                    
                    # --- 6. Target-Specific Metadata Snapshot (Event N) ---
                    "Target_Event_Idx", 
                    "Event_N_True_Sensor", 
                    "Event_N_Pred_Sensor",
                    "Event_N_Wavelet_Weights"
                ])

    def log_config_snapshot(self, config_module):
        """
        Saves a structured snapshot of hyperparameter dictionaries for research audit trails.
        """
        snapshot_path = os.path.join(self.output_dir, "config_snapshot.json")
        
        config_dict = {}
        for key in dir(config_module):
            if not key.startswith("__"):
                val = getattr(config_module, key)
                if isinstance(val, (int, float, str, bool, list, dict)):
                    config_dict[key] = val
                    
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Configuration parameter snapshot safely archived to '{snapshot_path}'")

    def save_training_history(self, train_history):
        """
        Saves step-by-step training and validation progress history to a CSV file.
        """
        history_path = os.path.join(self.output_dir, "training_history.csv")
        
        with open(history_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Epoch", "Total_Train_Loss", "Loc_Loss", "P_Loss",
                "Total_Val_Loss", "Val_Loc_Loss", "Val_P_Loss"
            ])
            
            for epoch, m in enumerate(train_history):
                writer.writerow([
                    epoch, m.get("total", 0.0), m.get("loc", 0.0), m.get("p", 0.0),
                    m.get("val_total", 0.0), m.get("val_loc", 0.0), m.get("val_p", 0.0)
                ])

    def write_csv_metrics(self, r):
        """
        Appends raw numerical records for plotting automation scripts.
        """
        idx = r['target_event_idx']
        
        with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                # --- 1. Global Core Intensity Head Metrics E(x) ---
                r['label'], r['e_loc_mean'], r['e_loc_std'], r['e_loc_median'], r['e_loc_normalized'],
                
                # --- 2. Global Core Frequency Head Metrics P(x) ---
                r['p_loc_mean'], r['p_loc_std'], r['p_loc_median'], r['p_loc_normalized'],
                r['peak_loc_mean'], r['peak_loc_std'], r['peak_loc_median'],
                # --- 3. Global Temporal Onset Metrics ---
                r['e_time_mean'], r['e_time_std'], r['e_time_p90'],
                
                # --- 4. Attribution & Explainability Metrics ---
                r['p_top1_acc'], r['p_top3_acc'], r['p_top5_acc'], r['p_top10_acc'], r['p_proximity_acc'],
                r['explainability_fidelity_delta'], r['sensor_dist_mean'], r['sensor_dist_std'], r['sensor_idx_error'],
                
                # --- 5. Explainability Profiles ---
                r['freq_score_mean'], r['freq_score_std'], r['freq_score_max'],
                r['temp_score_mean'], r['temp_score_std'], r['temp_score_max'],
                r['neigh_score_mean'], r['neigh_score_std'], r['neigh_score_max'],
                
                # --- 6. Event N Metadata Fields ---
                r['target_event_idx'], 
                r[f'event_{idx}_true_sensor'], 
                r[f'event_{idx}_pred_sensor'],
                str(r[f'event_{idx}_wavelet_weights'])
            ])

    def write_txt_report(self, r):
        """
        Generates a human-readable text document breaking down complete network profiles.
        """
        idx = r['target_event_idx']
        
        with open(self.txt_path, mode='a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"PERFORMANCE REPORT PROFILE: [{r['label']}]\n")
            f.write("-"*80 + "\n")
            f.write(f"• E(x) Mean Localization Error   : {r['e_loc_mean']:.4f} (std {r['e_loc_std']:.4f})\n")
            f.write(f"• E(x) Median Localization Error : {r['e_loc_median']:.4f} [ROBUST STATE]\n")
            f.write(f"• E(x) Normalized Grid Error     : {r['e_loc_normalized']:.4f}\n")
            f.write("-"*40 + "\n")
            f.write(f"• P(x) Mean Localization Error   : {r['p_loc_mean']:.4f} (std {r['p_loc_std']:.4f})\n")
            f.write(f"• P(x) Median Localization Error : {r['p_loc_median']:.4f} [ROBUST STATE]\n")
            f.write(f"• P(x) Normalized Grid Error     : {r['p_loc_normalized']:.4f}\n")
            f.write("-"*40 + "\n")
            f.write(f"• Peak Sensor Mean Loc Error     : {r['peak_loc_mean']:.4f} (std {r['peak_loc_std']:.4f})\n")
            f.write("-"*40 + "\n")
            f.write(f"• Mean Temporal Onset Error E(t) : {r['e_time_mean']:.4f} hours (std {r['e_time_std']:.4f})\n")
            f.write(f"• 90th Percentile Time Error     : {r['e_time_p90']:.4f} hours [BOUNDED TAIL]\n")
            f.write("-"*80 + "\n")
            f.write("EXPLAINABILITY STATISTICS\n")
            f.write(f"• P(x) Source Top-1 Accuracy     : {r['p_top1_acc']:.2f}%\n")
            f.write(f"• P(x) Source Top-3 Accuracy     : {r['p_top3_acc']:.2f}%\n")
            f.write(f"• P(x) Source Top-5 Accuracy     : {r['p_top5_acc']:.2f}%\n")
            f.write(f"• P(x) Source Top-10 Accuracy     : {r['p_top10_acc']:.2f}%\n")
            f.write(f"• P(x) Proximity Accuracy        : {r['p_proximity_acc']:.2f}%\n")
            f.write(f"• Explainability Fidelity Δ      : +{r['explainability_fidelity_delta']:.4f} degradation\n")
            f.write(f"• Continuous Sensor Distance     : {r['sensor_dist_mean']:.4f} (std {r['sensor_dist_std']:.4f})\n")
            f.write(f"• Discrete Index Error           : {r['sensor_idx_error']:.4f}\n")
            f.write("-"*80 + "\n")
            f.write("RAW ATTREBUTION METRICS\n")
            f.write(f"• Frequency Score Profile : mean={r['freq_score_mean']:.4f}, std={r['freq_score_std']:.4f}, max={r['freq_score_max']:.4f}\n")
            f.write(f"• Temporal Score Profile  : mean={r['temp_score_mean']:.4f}, std={r['temp_score_std']:.4f}, max={r['temp_score_max']:.4f}\n")
            f.write(f"• Neighbor Score Profile  : mean={r['neigh_score_mean']:.4f}, std={r['neigh_score_std']:.4f}, max={r['neigh_score_max']:.4f}\n")
            f.write("-"*80 + "\n")
            f.write(f"TARGET SPECIFIC SNAPSHOT (EVENT {idx})\n")
            f.write(f"• True Source Node Index  : {r[f'event_{idx}_true_sensor']}\n")
            f.write(f"• Predicted Source Index  : {r[f'event_{idx}_pred_sensor']}\n")
            f.write(f"• Wavelet Attention Vector: {r[f'event_{idx}_wavelet_weights']}\n")
            f.write("="*80 + "\n\n")
