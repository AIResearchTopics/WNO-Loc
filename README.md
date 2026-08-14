# WNO-Loc: Wavelet Neural Operator for Spatiotemporal Event Localization with Sensor-Level Explainability

IEEE Big Data 2026 Submission

## Overview

WNO-Loc is a spatiotemporal event localization framework combining 
per-feature graph spectral mixing with Wavelet Neural Operators (WNO). 
It simultaneously estimates:

- **E(x)** — geographic event location (km)
- **E(t)** — temporal onset (hours)
- **p** — origin-sensor distribution for localization explainability

Evaluated on 522 verified atmospheric events across 33 city-year 
combinations from US EPA and EU EEA regulatory networks (2022-2024), 
spanning 8 experimental folds covering temporal, spatial, and 
cross-continental transfer conditions.

## Dataset

The multi-continental atmospheric event dataset is available at:

[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1ygc_46GHGR6gmKWvsMRFpRo0mDNRCCAq?usp=drive_link)

Place the downloaded folder at `data/` in the repository root:

```
wno-loc/
├── data/
│   ├── EU/
│   │   ├── Berlin/
│   │   ├── Munich/
│   │   ├── Paris/
│   │   ├── Rome/
│   │   ├── Lisbon/
│   │   └── Stockholm/
│   └── US/
│       ├── Arizona/
│       ├── California/
│       ├── Nevada/
│       ├── NewMexico/
│       └── Utah/
├── model.py
├── main.py
├── config.py
└── ...
```

## Requirements

```bash
pip install torch pytorch_wavelets numpy pandas scikit-learn matplotlib
```

## Running Experiments

### Real-world data

```bash
python main.py \
    --fold TEMPORAL_EU \
    --real_data True \
    --datafolder data/
```

Available folds:

| Fold | Description |
|---|---|
| `TEMPORAL_EU` | Temporal generalization across EU cities |
| `US_TO_EUROPE` | Cross-continental large-to-small transfer |
| `WITHIN_EU` | Within-continent spatial transfer |
| `WITHIN_US_TEMPORAL` | Temporal transfer same network |
| `WITHIN_US_SPATIAL_S2M` | Small to medium scale transfer |
| `WITHIN_US_SPATIAL_L2M` | Large to medium scale transfer |
| `WITHIN_US_SPATIAL_M2L` | Medium to large scale transfer |
| `EUROPE_TO_US` | Cross-continental small-to-large transfer (negative control) |

### Synthetic data

```bash
python main.py --real_data False
```

### Evaluation only (from saved checkpoints)

```bash
python eval_only.py \
    --fold TEMPORAL_EU \
    --logs_path "path/to/run/directory"
```

### Aggregate multi-seed results

```bash
python aggregate_multi_seed.py \
    --fold TEMPORAL_EU \
    --logs_path "path/to/run/directory"
```

## Repository Structure

| File | Description |
|---|---|
| `model.py` | WNO-Loc model — graph spectral layer, WNO, P(x) head |
| `main.py` | Training and evaluation pipeline |
| `config.py` | Hyperparameter configuration |
| `evaluate.py` | Evaluation metrics — E(x), E(t), P(x) |
| `eval_only.py` | Evaluation from saved checkpoints |
| `real_dataset_loader.py` | Dataset loading and fold construction |
| `baseline_lstm.py` | BiLSTM baseline |
| `baseline_fno.py` | FNO baseline |
| `baseline_transformer.py` | Transformer baseline |
| `baseline_idw.py` | IDW baseline |
| `synthetic_data.py` | Synthetic event generator |
| `aggregate_multi_seed.py` | Multi-seed result aggregation |
| `ablation_study.py` | Ablation experiments |

## Key Results

### European Folds (point-source events, dense networks)

| Fold | E(x) Test (km) | Proximity Accuracy Test |
|---|---|---|
| Temporal-EU | 7.90±0.29 | 89.5% |
| US→EU | 9.33±1.11 | 73.0% |
| Within-EU | 9.48±0.14 | 84.2% |

### US Folds (area-source events, sparse networks)

| Fold | E(x) Test (km) | Proximity Accuracy Test |
|---|---|---|
| Within-US-S2M | 166.1±5.0 | 9.7% |
| Within-US-L2M | 157.3±4.1 | 9.7% |
| Within-US-Temporal | 165.1±7.2 | 5.6% |
| Within-US-M2L | 312.6±22.4 | 1.5% |
| EU→US | 353.3±39.2 | 2.8% |

US proximity values exceed random baseline (1/N = 1--3%) confirming 
discriminative signal persists under sparse sensing.

## Hyperparameters

| Parameter | Value |
|---|---|
| WNO width | 32 |
| WNO levels | 3 |
| WNO layers | 2 |
| Wavelet | db4 |
| Graph K | 3 (Chebyshev order) |
| k neighbors | 4 |
| Learning rate | 5e-4 |
| Epochs | 120 |
| Batch size | 1 (real), 8 (synthetic) |
| λ_p | 0.1 (<200 events), 0.5 (≥200 events) |
| λ_e | 0.01 (real data only) |
| Contrastive weight | 0.1 (real data only) |

## Citation

```bibtex
@inproceedings{2026wnoloc,
  title     = {WNO-Loc: Wavelet Neural Operator for Spatiotemporal
               Event Localization with Sensor-Level Explainability},
  booktitle = {2026 IEEE International Conference on Big Data},
  year      = {2026}
}
```

## License

MIT License
