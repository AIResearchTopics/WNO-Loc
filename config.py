# -*- coding: utf-8 -*-
"""
Global Configuration Settings for Spatiotemporal Localization Framework.
Optimized for experimental reproducibility at IEEE Big Data 2026.

Usage from command line:
    python main.py --seed 42 --fold WITHIN_EU --fast_mode False --epochs 200
    python main.py --seed 44 --fold TEMPORAL --lambda_p 0.1 --real_data True

All parameters have defaults defined here. Command-line arguments
override these defaults when provided.

@author: usman.anjum
"""

import argparse
import os as _os
import torch
from datetime import datetime


# ---------------------------------------------------------------------------
# Default values -- edit these for your environment
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # Paths
    'datafolder':   r"/content/drive/MyDrive/WNO Project/Datafolder",

    # Experiment
    'seed':         40,
    'fold':         'US_TO_EUROPE',
    'real_data':    True,
    'force_reprocess': False,

    # Data
    'n_sensors':    20,
    'T':            100,
    'n_features':   3,

    # Model
    'fast_mode':    True,

    # Training
    'n_epochs':     120,
    'display_epoch': 20,
    'lr':           5e-4,
    'scheduler_step': 50,
    'scheduler_gamma': 0.5,
    'lambda_p':     0.5,
    'lambda_c':     0.0,
}


# ---------------------------------------------------------------------------
# Argument parser -- allows command-line overrides
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description='WNO Spatiotemporal Event Localization Framework'
    )

    # Paths
    parser.add_argument('--datafolder', type=str,
                        default=_DEFAULTS['datafolder'],
                        help='Root data folder containing EU/ and US/ subfolders')

    # Experiment
    parser.add_argument('--seed', type=int,
                        default=int(_os.environ.get('EXP_SEED',
                                                     _DEFAULTS['seed'])),
                        help='Random seed (also set via EXP_SEED env var)')
    parser.add_argument('--fold', type=str, default=_DEFAULTS['fold'],
                        choices=[
                            'US_TO_EUROPE',
                            'WITHIN_EU',
                            'WITHIN_US_TEMPORAL',
                            'WITHIN_US_SPATIAL_S2M',
                            'WITHIN_US_SPATIAL_M2L',
                            'WITHIN_US_SPATIAL_L2M',
                            'TEMPORAL_EU',
                            'EUROPE_TO_US',
                        ], help='Data fold / split strategy')
    parser.add_argument('--real_data',
                        type=lambda x: x.lower() == 'true',
                        default=_DEFAULTS['real_data'],
                        help='True = real data, False = synthetic')
    parser.add_argument('--force_reprocess',
                        type=lambda x: x.lower() == 'true',
                        default=_DEFAULTS['force_reprocess'],
                        help='Force reprocessing of raw CSVs (ignores cache)')

    # Data
    parser.add_argument('--n_sensors',  type=int,
                        default=_DEFAULTS['n_sensors'],
                        help='Number of sensors (synthetic data only)')
    parser.add_argument('--T',          type=int,
                        default=_DEFAULTS['T'],
                        help='Time window length in hours')
    parser.add_argument('--n_features', type=int,
                        default=_DEFAULTS['n_features'],
                        help='Number of features (synthetic data only)')

    # Model
    parser.add_argument('--fast_mode',
                        type=lambda x: x.lower() == 'true',
                        default=_DEFAULTS['fast_mode'],
                        help='True = small fast model, False = full model')

    # Training
    parser.add_argument('--n_epochs',       type=int,
                        default=_DEFAULTS['n_epochs'])
    parser.add_argument('--display_epoch',  type=int,
                        default=_DEFAULTS['display_epoch'])
    parser.add_argument('--lr',             type=float,
                        default=_DEFAULTS['lr'])
    parser.add_argument('--scheduler_step', type=int,
                        default=_DEFAULTS['scheduler_step'])
    parser.add_argument('--scheduler_gamma',type=float,
                        default=_DEFAULTS['scheduler_gamma'])
    parser.add_argument('--lambda_p',       type=float,
                        default=_DEFAULTS['lambda_p'],
                        help='Weight on P(x) proximity loss')
    parser.add_argument('--lambda_c',       type=float,
                        default=_DEFAULTS['lambda_c'],
                        help='Weight on consistency loss (0 = disabled)')

    parser.add_argument('--no_display', action='store_true', default=False,
                    help='Disable all plot display (use Agg backend)')
                    
    # parse_known_args so Jupyter/Colab extra args don't crash
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Parse arguments and expose as module-level constants
# ---------------------------------------------------------------------------
_args = _parse_args()

# Paths
DATAFOLDER = _args.datafolder

# Experiment
SEED            = _args.seed
DATA_SELECTOR   = _args.fold       # single string now, not a list
USE_REAL_DATA   = _args.real_data
FORCE_REPROCESS = _args.force_reprocess

# Data
N_SENSORS  = _args.n_sensors
T          = _args.T
N_FEATURES = _args.n_features

# Model
FAST_MODE = _args.fast_mode
if FAST_MODE:
    WNO_CONFIG = dict(wno_width=32, wno_levels=3,
                      wno_layers=2, wavelet='db4')
else:
    WNO_CONFIG = dict(wno_width=32, wno_levels=3,
                      wno_layers=2, wavelet='sym4')

# Training
N_EPOCHS        = _args.n_epochs
DISPLAY_EPOCH   = _args.display_epoch
LEARNING_RATE   = _args.lr
SCHEDULER_STEP  = _args.scheduler_step
SCHEDULER_GAMMA = _args.scheduler_gamma
LAMBDA_P        = _args.lambda_p
LAMBDA_C        = _args.lambda_c

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        name   = 'Intel Arc / Intel XPU'
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        name   = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        name   = 'Apple Metal'
    else:
        device = torch.device('cpu')
        name   = 'CPU'
    print(f'\nUsing device: {name}')
    return device

DEVICE = get_device()

# ---------------------------------------------------------------------------
# Run ID and output directory
# ---------------------------------------------------------------------------
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')
RUN_ID       = _os.environ.get(
    'EXP_RUN_ID',
    datetime.now().strftime('%Y-%m-%d_%H%M%S')
)
# Detect if running on Kaggle
_on_kaggle = _os.path.exists('/kaggle/working')

if _on_kaggle:
    OUTPUT_DIR = f'/kaggle/working/logs/{RUN_ID}/seed{SEED}'
else:
    OUTPUT_DIR = f'logs/{RUN_ID}/seed{SEED}'


# ---------------------------------------------------------------------------
# Print active configuration -- called from main.py at startup
# ---------------------------------------------------------------------------
def print_config():
    print('\n' + '='*60)
    print('CONFIGURATION')
    print('='*60)
    print(f'  Seed:           {SEED}')
    print(f'  Fold:           {DATA_SELECTOR}')
    print(f'  Real data:      {USE_REAL_DATA}')
    print(f'  Force reprocess:{FORCE_REPROCESS}')
    print(f'  FAST_MODE:      {FAST_MODE}')
    print(f'  WNO config:     {WNO_CONFIG}')
    print(f'  T:              {T}')
    print(f'  N_epochs:       {N_EPOCHS}')
    print(f'  LR:             {LEARNING_RATE}')
    print(f'  Lambda_p:       {LAMBDA_P}')
    print(f'  Lambda_c:       {LAMBDA_C}')
    print(f'  Output dir:     {OUTPUT_DIR}')
    print(f'  Data folder:    {DATAFOLDER}')
    print('='*60 + '\n')