#!/usr/bin/env python3
"""
Random-search Optuna tuner for HierLocal on 20 agents.

This script samples hyperparameters from user-defined ranges, launches
`main.py` as a subprocess, then reads the saved run log to score each trial.

Typical usage:
    python3 scripts/optuna_hierlocal_random_search.py --trials 20 --trial-epochs 400

The best configuration is written to:
    logs/optuna_hierlocal/best_params.json
    logs/optuna_hierlocal/trials.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import shlex
from pathlib import Path

import optuna
from optuna.samplers import RandomSampler

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(ROOT, 'logs', 'optuna_hierlocal')
os.makedirs(LOG_DIR, exist_ok=True)


def extract_success(log_path):
    if not os.path.exists(log_path):
        return None
    try:
        data = torch.load(log_path, weights_only=False)
    except Exception:
        try:
            data = torch.load(log_path)
        except Exception:
            return None

    log = data.get('log', data)
    for key in ('success', 'eval_success'):
        if key in log and hasattr(log[key], 'data'):
            arr = list(log[key].data)
            if len(arr) > 0:
                return float(arr[-1])

    if 'reward' in log and hasattr(log['reward'], 'data'):
        arr = list(log['reward'].data)
        if len(arr) > 0:
            return float(arr[-1])

    return None


def suggested_team_sizes(nagents):
    upper = max(3, nagents // 2)
    values = [3, 4, 5, 6, 8, 10]
    return [v for v in values if v <= upper] or [max(2, upper)]


def build_cmd(args, trial, params, trial_seed):
    save_path = os.path.join(LOG_DIR, f"trial_{trial.number:04d}_seed{trial_seed}.pt")
    log_path = os.path.join(LOG_DIR, f"trial_{trial.number:04d}_seed{trial_seed}_log.pt")

    cmd = [
        sys.executable, os.path.join(ROOT, 'main.py'),
        '--env_name', args.env_name,
        '--nagents', str(args.nagents),
        '--add_rate_min', str(args.add_rate_min),
        '--add_rate_max', str(args.add_rate_max),
        '--nprocesses', '1',
        '--epoch_size', str(args.epoch_size),
        '--batch_size', str(params['batch_size']),
        '--hid_size', str(params['hid_size']),
        '--detach_gap', str(args.detach_gap),
        '--lrate', str(params['lrate']),
        '--dim', str(args.dim),
        '--max_steps', str(args.max_steps),
        '--ic3net',
        '--vision', str(args.vision),
        '--comm_passes', str(params['comm_passes']),
        '--difficulty', args.difficulty,
        '--crash_penalty', str(args.crash_penalty),
        '--terminal_reward', str(args.terminal_reward),
        '--num_epochs', str(args.trial_epochs),
        '--entr', str(params['entr']),
        '--seed', str(trial_seed),
        '--hier_local_attn',
        '--team_size', str(params['team_size']),
        '--max_grad_norm', str(params['max_grad_norm']),
        '--save', save_path,
        '--log_path', log_path,
    ]

    return cmd, save_path, log_path


def run_trial(args, trial):
    params = {
        'hid_size': trial.suggest_categorical('hid_size', args.hid_sizes),
        'lrate': trial.suggest_float('lrate', args.lrate_min, args.lrate_max, log=True),
        'entr': trial.suggest_float('entr', args.entr_min, args.entr_max, log=True),
        'batch_size': trial.suggest_categorical('batch_size', args.batch_sizes),
        'comm_passes': trial.suggest_int('comm_passes', args.comm_passes_min, args.comm_passes_max),
        'team_size': trial.suggest_categorical('team_size', args.team_sizes),
        'max_grad_norm': trial.suggest_float('max_grad_norm', args.grad_min, args.grad_max),
    }

    trial_seed = trial.suggest_categorical('seed', args.seeds)
    cmd, save_path, log_path = build_cmd(args, trial, params, trial_seed)
    cmd_str = ' '.join(shlex.quote(part) for part in cmd)

    stdout_path = os.path.join(LOG_DIR, f"trial_{trial.number:04d}_stdout.txt")
    stderr_path = os.path.join(LOG_DIR, f"trial_{trial.number:04d}_stderr.txt")

    start = time.time()
    with open(stdout_path, 'wb') as out_f, open(stderr_path, 'wb') as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, check=False)
    duration = time.time() - start

    success = extract_success(log_path)
    if success is None:
        success = -1.0

    row = {
        'trial': trial.number,
        'seed': trial_seed,
        'cmd': cmd_str,
        'exit_code': proc.returncode,
        'stdout': stdout_path,
        'stderr': stderr_path,
        'save_path': save_path,
        'log_path': log_path,
        'duration': duration,
        'final_success': success,
        **params,
    }
    return success, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--trial-epochs', type=int, default=400)
    parser.add_argument('--env-name', default='traffic_junction')
    parser.add_argument('--nagents', type=int, default=20)
    parser.add_argument('--add-rate-min', type=float, default=1.0)
    parser.add_argument('--add-rate-max', type=float, default=1.0)
    parser.add_argument('--epoch-size', type=int, default=20)
    parser.add_argument('--detach-gap', type=int, default=10)
    parser.add_argument('--dim', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--vision', type=int, default=2)
    parser.add_argument('--difficulty', default='medium')
    parser.add_argument('--crash-penalty', type=float, default=-15)
    parser.add_argument('--terminal-reward', type=float, default=15)
    parser.add_argument('--hid-sizes', type=int, nargs='+', default=[64, 128, 256])
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[64, 128, 150, 256, 500])
    parser.add_argument('--team-sizes', type=int, nargs='+', default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--lrate-min', type=float, default=1e-5)
    parser.add_argument('--lrate-max', type=float, default=1e-3)
    parser.add_argument('--entr-min', type=float, default=1e-6)
    parser.add_argument('--entr-max', type=float, default=1e-3)
    parser.add_argument('--comm-passes-min', type=int, default=1)
    parser.add_argument('--comm-passes-max', type=int, default=3)
    parser.add_argument('--grad-min', type=float, default=0.5)
    parser.add_argument('--grad-max', type=float, default=2.0)
    parser.add_argument('--sampler-seed', type=int, default=42)
    parser.add_argument('--out-csv', default=os.path.join(LOG_DIR, 'trials.csv'))
    parser.add_argument('--out-best', default=os.path.join(LOG_DIR, 'best_params.json'))
    parser.add_argument('--study-name', default='hierlocal_random_search')
    args = parser.parse_args()

    if args.team_sizes is None:
        args.team_sizes = suggested_team_sizes(args.nagents)

    rows = []

    sampler = RandomSampler(seed=args.sampler_seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=args.study_name)

    def objective(trial):
        score, row = run_trial(args, trial)
        rows.append(row)
        return score

    study.optimize(objective, n_trials=args.trials)

    with open(args.out_csv, 'w', newline='') as f:
        fieldnames = [
            'trial', 'seed', 'hid_size', 'lrate', 'entr', 'batch_size',
            'comm_passes', 'team_size', 'max_grad_norm', 'exit_code',
            'stdout', 'stderr', 'save_path', 'log_path', 'duration',
            'final_success', 'cmd',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    best_payload = {
        'best_value': study.best_value,
        'best_trial_number': study.best_trial.number,
        'best_params': study.best_trial.params,
        'trials_csv': args.out_csv,
    }
    with open(args.out_best, 'w') as f:
        json.dump(best_payload, f, indent=2)

    print('Best value:', study.best_value)
    print('Best trial:', study.best_trial.number)
    print('Best params:', json.dumps(study.best_trial.params, indent=2))
    print('Wrote trial CSV to', args.out_csv)
    print('Wrote best params to', args.out_best)


if __name__ == '__main__':
    main()