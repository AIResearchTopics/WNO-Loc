# WNO Spatiotemporal Event Localization — Running Guide

## Overview

Three scripts form the complete experimental pipeline:

```
main.py                 <- single training + evaluation run
multi_seed_run.py       <- runs main.py across 5 seeds automatically
aggregate_multi_seed_run.py  <- aggregates results from multi_seed_run.py
```

---

## 1. main.py — Single Run

Trains the WNO localization model and evaluates against all baselines.

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--datafolder` | str | config default | Path to root data folder containing EU/ and US/ |
| `--seed` | int | 40 | Random seed |
| `--fold` | str | US_TO_EUROPE | Data fold (see folds below) |
| `--real_data` | bool | True | True = real data, False = synthetic |
| `--force_reprocess` | bool | False | Rebuild data cache from raw CSVs |
| `--n_epochs` | int | 120 | Number of training epochs |
| `--display_epoch` | int | 20 | Print validation every N epochs |
| `--lr` | float | 5e-4 | Learning rate |
| `--lambda_p` | float | 0.5 | Weight on P(x) proximity loss |
| `--lambda_c` | float | 0.0 | Weight on consistency loss (0 = disabled) |
| `--fast_mode` | bool | True | True = small fast model, False = full model |
| `--T` | int | 100 | Time window length in hours |
| `--n_sensors` | int | 20 | Number of sensors (synthetic only) |
| `--n_features` | int | 3 | Number of features (synthetic only) |

### Available Folds

| Fold | Train | Val | Test |
|---|---|---|---|
| `US_TO_EUROPE` | All US cities (382 events) | Paris + Munich (52 events) | Berlin + Rome (46 events) |
| `EUROPE_TO_US` | All EU cities (140 events) | Arizona + Nevada | California + NM + Utah |
| `WITHIN_EU` | Munich + Rome + Lisbon + Stockholm (93 events) | Paris (23 events) | Berlin (24 events) |
| `WITHIN_US` | CA + NV + NM + UT | Arizona 2022 (75 events) | Arizona 2023+2024 (103 events) |
| `TEMPORAL` | All cities 2022 (196 events) | All cities 2023 (155 events) | All cities 2024 (171 events) |

### Examples

```bash
# Real data, WITHIN_EU fold, single seed
python main.py --fold WITHIN_EU --seed 42 --real_data True

# Real data, US_TO_EUROPE, 60 epochs
python main.py --fold US_TO_EUROPE --seed 40 --n_epochs 60

# Synthetic data
python main.py --real_data False --seed 42 --n_epochs 120

# Force rebuild data cache
python main.py --fold WITHIN_EU --force_reprocess True

# Custom data folder
python main.py --fold US_TO_EUROPE --datafolder "/path/to/data"

# Full model (slower, better)
python main.py --fold WITHIN_EU --fast_mode False --n_epochs 200
```

### Output

Results are saved to `logs/{RUN_ID}/seed{SEED}/`:
```
logs/
  2026-07-31_120000/
    seed40/
      evaluation_report.txt       <- human-readable results
      evaluation_metrics.csv      <- machine-readable metrics
      training_history.csv        <- loss curves
      config_snapshot.json        <- exact config used
      best_checkpoint.pt          <- best model weights
      plots/                      <- all figures
      explainability_plots/       <- causal curves, attribution plots
```

---

## 2. multi_seed_run.py — Multi-Seed Sweep

Runs `main.py` automatically for each seed. Skips seeds that already
have results (safe to resume after interruption).

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--fold` | str | US_TO_EUROPE | Data fold to run |
| `--datafolder` | str | None | Override data folder path |
| `--seeds` | int list | 40 41 42 43 44 | Seeds to run |
| `--run_id` | str | auto timestamp | Resume an existing run by ID |
| `--real_data` | bool | True | True = real data, False = synthetic |
| `--n_epochs` | int | 120 | Passed to main.py |

### Examples

```bash
# Run full 5-seed sweep, WITHIN_EU
python multi_seed_run.py --fold WITHIN_EU

# Run specific fold with custom data folder
python multi_seed_run.py --fold US_TO_EUROPE --datafolder "/path/to/data"

# Run only 3 seeds
python multi_seed_run.py --fold WITHIN_US --seeds 40 41 42

# Resume interrupted sweep (skips completed seeds automatically)
python multi_seed_run.py --fold WITHIN_EU --run_id 2026-07-31_120000

# Synthetic data sweep
python multi_seed_run.py --fold US_TO_EUROPE --real_data False --n_epochs 120

# Fewer epochs for faster run
python multi_seed_run.py --fold TEMPORAL --n_epochs 60
```

### On Kaggle

```python
!python multi_seed_run.py \
    --fold WITHIN_EU \
    --datafolder /kaggle/input/datasets/usmananjum/wno-main-dataset/Datafolder \
    --real_data True \
    --n_epochs 120
```

### Resume after timeout / interruption

```python
# Pass the existing RUN_ID -- completed seeds are skipped
!python multi_seed_run.py \
    --fold WITHIN_EU \
    --run_id 2026-07-31_120000 \
    --datafolder /kaggle/input/datasets/usmananjum/wno-main-dataset/Datafolder
```

---

## 3. aggregate_multi_seed_run.py — Results Aggregation

Reads all seed results from a completed multi-seed sweep and produces:
- Summary table (mean ± std per metric per model)
- Statistical tests (paired t-test, Cohen's d vs all baselines)
- Boxplot figures

**Run this AFTER multi_seed_run.py has completed all seeds.**

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--run_id` | str | None | RUN_ID to analyze (skips interactive prompt) |

### Examples

```bash
# Interactive mode -- lists all available runs, you choose
python aggregate_multi_seed_run.py

# Direct mode -- pass RUN_ID directly
python aggregate_multi_seed_run.py --run_id 2026-07-31_120000
```

### On Kaggle

```python
!python aggregate_multi_seed_run.py --run_id 2026-07-31_120000
```

### Output

Results saved to `logs/{RUN_ID}/analysis/`:
```
logs/
  2026-07-31_120000/
    analysis/
      multiseed_summary.csv       <- mean ± std per metric per model
      statistical_tests.csv       <- t-test p-values and Cohen's d
      all_seed_data.csv           <- raw per-seed per-metric data
      Ex_Loc_Mean_comparison.png  <- boxplot figures
      Px_Top5_Acc_comparison.png
      Px_Proximity_Acc_comparison.png
```

---

## Complete Workflow Example

### Local (Windows)

```bash
# Step 1: Run 5-seed sweep for WITHIN_EU
python multi_seed_run.py --fold WITHIN_EU --real_data True --n_epochs 120

# Step 2: Aggregate results (run after sweep completes)
python aggregate_multi_seed_run.py --run_id 2026-07-31_120000

# Step 3: Run next fold
python multi_seed_run.py --fold US_TO_EUROPE --real_data True --n_epochs 120
python aggregate_multi_seed_run.py --run_id 2026-07-31_150000
```

### Kaggle

```python
# Cell 1: Install dependencies
!pip install pytorch-wavelets -q

# Cell 2: Run sweep
!python multi_seed_run.py \
    --fold US_TO_EUROPE \
    --datafolder /kaggle/input/datasets/usmananjum/wno-main-dataset/Datafolder \
    --real_data True \
    --n_epochs 120

# Cell 3: Aggregate (after sweep completes)
!python aggregate_multi_seed_run.py --run_id 2026-07-31_120000
```

---

## Data Folder Structure

```
Datafolder/
    EU/
        Berlin/
            Berlin_2022.csv            <- sensor data (PM2.5, PM10, NO2, ...)
            Berlin_2023.csv
            Berlin_2024.csv
            Berlin_events_2022.csv     <- event data (lat, lon, date, type)
            Berlin_events_2023.csv
            Berlin_events_2024.csv
        Lisbon/   Munich/   Paris/   Rome/   Stockholm/
    US/
        Arizona/
            Arizona_2022.csv
            Arizona_2023.csv
            Arizona_events_2022.csv
            Arizona_events_2023.csv
        California/   Nevada/   NewMexico/   Utah/
    .cache/
        regions_XXXX.npz              <- auto-generated, speeds up loading
```

The `.cache/` folder is created automatically on first run. Delete it
or set `--force_reprocess True` to rebuild when adding new data.

---

## Environment Variables

These are set automatically by `multi_seed_run.py` and should not
be set manually:

| Variable | Description |
|---|---|
| `EXP_SEED` | Current seed (read by config.py) |
| `EXP_RUN_ID` | Current run ID (read by config.py) |
| `MPLBACKEND` | Set to `Agg` to prevent plot popups |

---

## Troubleshooting

**Code stops at plot / display window:**
matplotlib is trying to show an interactive window.
Fix: ensure `MPLBACKEND=Agg` is set, or run via `multi_seed_run.py`
which sets this automatically.

**Read-only file system error on Kaggle:**
Logs are trying to write to the input directory.
Fix: config.py auto-detects Kaggle and writes to `/kaggle/working/logs/`.

**Cache not found / reprocessing every run:**
The `.cache/` folder is inside `DATAFOLDER`. On Kaggle the input
directory is read-only so the cache cannot be saved there.
Fix: copy data to `/kaggle/working/` first, or pre-generate the cache
locally and include it in your uploaded dataset.

**Unknown fold error:**
`DATA_SELECTOR` is being read as a single character.
Fix: ensure `main.py` line 88 reads `config.DATA_SELECTOR` not
`config.DATA_SELECTOR[0]`.

**Graph construction is slow:**
First run always builds the sensor graph (expensive for CA: 227 sensors).
After first run it is cached. Set `--force_reprocess False` on all
subsequent runs to use the cache.
