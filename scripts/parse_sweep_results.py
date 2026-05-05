#!/usr/bin/env python3
"""
Parse sweep results and extract final success metric from run logs.

Usage:
    python3 scripts/parse_sweep_results.py --csv logs/sweep_team_size/results.csv

If --csv omitted, searches logs/sweep_team_size/results.csv by default.
Outputs summary CSV with columns: team_size, seed, final_success
"""
import argparse
import os
import csv
import torch
from pathlib import Path


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
    # handle wrapped dict
    log = data.get('log', data)
    # success may be under 'success' or 'eval_success'
    if 'success' in log and hasattr(log['success'], 'data'):
        arr = list(log['success'].data)
        if len(arr) > 0:
            return float(arr[-1])
    if 'eval_success' in log and hasattr(log['eval_success'], 'data'):
        arr = list(log['eval_success'].data)
        if len(arr) > 0:
            return float(arr[-1])
    # fallback: try 'rewards'
    if 'reward' in log and hasattr(log['reward'], 'data'):
        arr = list(log['reward'].data)
        if len(arr) > 0:
            return float(arr[-1])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='logs/sweep_team_size/results.csv')
    parser.add_argument('--out', default='logs/sweep_team_size/summary.csv')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv)
        return

    rows = []
    with open(args.csv, 'r') as f:
        r = csv.DictReader(f)
        for rec in r:
            team_size = rec.get('team_size')
            seed = rec.get('seed')
            log_path = rec.get('stdout') or rec.get('log_path') or rec.get('cmd')
            # attempt to find run_log in the command
            found = None
            if log_path and isinstance(log_path, str):
                # look for run_log_*.pt in the command string
                import re
                m = re.search(r'run_log_[^\s\'\"]+\.pt', log_path)
                if m:
                    found = m.group(0)
            # else try glob
            if not found:
                # look for run_log files in project root matching team_size and seed
                base = Path('')
                candidates = list(base.glob(f'run_log_*k{team_size}*_s{seed}*.pt'))
                if candidates:
                    found = str(candidates[0])
                else:
                    candidates = list(base.glob(f'run_log_*k{team_size}*.pt'))
                    if candidates:
                        found = str(candidates[0])
            if not found:
                # try the common pattern
                fname = f'run_log_20agents_HierLocal_k{team_size}_s{seed}.pt'
                if os.path.exists(fname):
                    found = fname
            success = None
            if found:
                success = extract_success(found)
            rows.append({'team_size': team_size, 'seed': seed, 'log_file': found or '', 'final_success': success})

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['team_size','seed','log_file','final_success'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('Wrote summary to', args.out)

if __name__ == '__main__':
    main()
