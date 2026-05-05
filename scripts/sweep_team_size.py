#!/usr/bin/env python3
"""
Sweep `team_size` values by launching `main.py` with `--hier_local_attn`.
Saves stdout/stderr to `logs/` and writes a CSV `results.csv` with metadata.

Usage:
    python3 scripts/sweep_team_size.py --team-sizes 4 6 8 10 --seeds 1 2 --dry-run

Set --dry-run to only print commands without executing.
"""
import argparse
import subprocess
import shlex
import sys
import os
import csv
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(ROOT, 'logs', 'sweep_team_size')
os.makedirs(LOG_DIR, exist_ok=True)

def build_cmd(team_size, seed, extra_args):
    cmd = [
        sys.executable, os.path.join(ROOT, 'main.py'),
        '--env_name', 'traffic_junction',
        '--nagents', '20',
        '--hier_local_attn',
        '--team_size', str(team_size),
        '--hid_size', '128',
        '--comm_passes', '1',
        '--lrate', '5e-5',
        '--entr', '5e-5',
        '--batch_size', '500',
        '--num_epochs', '400',
        '--seed', str(seed),
        '--save', os.path.join(ROOT, f'model_20agents_HierLocal_k{team_size}_s{seed}.pt'),
        '--log_path', os.path.join(ROOT, f'run_log_20agents_HierLocal_k{team_size}_s{seed}.pt')
    ]
    cmd += extra_args
    return cmd


def run_one(team_size, seed, dry_run=False, extra_args=[]):
    cmd = build_cmd(team_size, seed, extra_args)
    cmd_str = ' '.join(shlex.quote(c) for c in cmd)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    stdout_path = os.path.join(LOG_DIR, f'k{team_size}_s{seed}_{timestamp}.out')
    stderr_path = os.path.join(LOG_DIR, f'k{team_size}_s{seed}_{timestamp}.err')

    if dry_run:
        print('[DRY RUN]', cmd_str)
        return {'team_size': team_size, 'seed': seed, 'cmd': cmd_str, 'exit_code': None, 'stdout': None, 'stderr': None, 'duration': 0.0}

    print('Running:', cmd_str)
    start = time.time()
    with open(stdout_path, 'wb') as out_f, open(stderr_path, 'wb') as err_f:
        p = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
        exit_code = p.wait()
    duration = time.time() - start
    return {'team_size': team_size, 'seed': seed, 'cmd': cmd_str, 'exit_code': exit_code,
            'stdout': stdout_path, 'stderr': stderr_path, 'duration': duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-sizes', type=int, nargs='+', default=[4,6,8,10])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--dry-run', action='store_true', default=False)
    parser.add_argument('--extra-args', type=str, nargs='*', default=[],
                        help='Extra args appended to every command')
    parser.add_argument('--out-csv', type=str, default=os.path.join(LOG_DIR, 'results.csv'))
    args = parser.parse_args()

    rows = []
    for k in args.team_sizes:
        for s in args.seeds:
            r = run_one(k, s, dry_run=args.dry_run, extra_args=args.extra_args)
            rows.append(r)

    # write CSV
    with open(args.out_csv, 'w', newline='') as csvfile:
        fieldnames = ['team_size','seed','cmd','exit_code','stdout','stderr','duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print('Sweep finished. Results written to', args.out_csv)

if __name__ == '__main__':
    main()
