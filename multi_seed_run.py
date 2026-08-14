# -*- coding: utf-8 -*-
import subprocess
import os
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=str, default='US_TO_EUROPE',
                    choices=[
                        'US_TO_EUROPE',
                        'WITHIN_EU',
                        'WITHIN_US_TEMPORAL',
                        'WITHIN_US_SPATIAL_S2M',
                        'WITHIN_US_SPATIAL_M2L',
                        'WITHIN_US_SPATIAL_L2M',
                        'TEMPORAL_EU',
                        'EUROPE_TO_US',
                    ])
parser.add_argument('--run_id',   type=str, default=None,
                    help='Resume existing run ID (omit to start new run)')
parser.add_argument('--seeds',    type=int, nargs='+', default=[40,41,42,43,44],
                    help='Seeds to run e.g. --seeds 40 41 42')
parser.add_argument('--real_data',type=lambda x: x.lower()=='true',
                    default=True)
parser.add_argument('--lambda_p', type=float, default=0.5,
                    help='Weight on P(x) proximity loss')
parser.add_argument('--lambda_c', type=float, default=0.0,
                    help='Weight on consistency loss')
parser.add_argument('--fast_mode',
                        type=lambda x: x.lower() == 'true',
                        default=True,
                        help='True = small fast model, False = full model')
args = parser.parse_args()

FOLD     = args.fold
SEEDS    = args.seeds
RUN_ID   = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")

print(f"Sweep run ID: {RUN_ID}")
print(f"Fold:         {FOLD}")
print(f"Seeds:        {SEEDS}")

for seed in SEEDS:
    log_dir    = f"logs/{RUN_ID}/seed{seed}"
    out_marker = os.path.join(log_dir, "evaluation_metrics.csv")

    if os.path.exists(out_marker):
        print(f"Seed {seed} already has results, skipping.")
        continue

    print(f"\n{'='*60}\nRunning seed {seed}\n{'='*60}")
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env["EXP_SEED"]   = str(seed)
    env["EXP_RUN_ID"] = RUN_ID
    env["MPLBACKEND"] = "Agg"

    cmd = (f"python -u main.py "
           f"--fold {FOLD} "
           f"--seed {seed} "
           f"--lambda_p {args.lambda_p} "
           f"--lambda_c {args.lambda_c} "
           f"--fast_mode {args.fast_mode} "
           f"--real_data {args.real_data}")
    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode != 0:
        print(f"WARNING: seed {seed} exited with code {result.returncode}")