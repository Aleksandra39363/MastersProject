#!/usr/bin/env python
"""
Visualize training progress from saved logs
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid PIL/Tkinter issues
import matplotlib.pyplot as plt
import traceback
import torch
import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

def _smooth_series(values, epochs, window):
    if values is None or epochs is None:
        return None, None
    if window <= 1 or len(values) < window:
        return values, epochs
    kernel = np.ones(window) / float(window)
    smoothed_valid = np.convolve(values, kernel, mode='valid')
    pad_length = window - 1 
    #epochs_smoothed = epochs[window - 1:]
    #return smoothed, epochs_smoothed
    smoothed = np.concatenate([np.full(pad_length, smoothed_valid[0]), smoothed_valid])  # Pad the beginning with original values
   
    return smoothed, epochs


def _prepare_group_metric(group_data, metric_key, smoothing_window):
    processed = []
    max_len = 0
    epoch_ref = None

    for data in group_data:
        if metric_key not in data or 'epochs' not in data:
            continue
        values = data[metric_key]
        if values is None:
            continue

        window = min(smoothing_window, len(values)) if len(values) > 1 else 1
        smoothed, epochs_smoothed = _smooth_series(values, data['epochs'], window)
        if smoothed is None or epochs_smoothed is None:
            continue

        smoothed = np.asarray(smoothed, dtype=float)
        epochs_smoothed = np.asarray(epochs_smoothed)
        processed.append((epochs_smoothed, smoothed))

        if len(smoothed) > max_len:
            max_len = len(smoothed)
            epoch_ref = epochs_smoothed

    if not processed:
        return None, None

    stacked = np.full((len(processed), max_len), np.nan, dtype=float)
    for i, (_, smoothed) in enumerate(processed):
        stacked[i, :len(smoothed)] = smoothed

    mean_series = np.nanmean(stacked, axis=0)
    valid = ~np.isnan(mean_series)
    return processed, (epoch_ref[valid], mean_series[valid])


def _to_float_array(values):
    out = []
    for val in values:
        if isinstance(val, np.ndarray):
            out.append(float(np.mean(val)) if val.size > 0 else 0.0)
        else:
            out.append(float(val))
    return np.asarray(out, dtype=float)


def _prepare_group_eval(group_data, metric_key):
    processed = []
    epoch_to_values = defaultdict(list)

    for data in group_data:
        eval_epochs = data.get('eval_epochs')
        eval_values = data.get(metric_key)
        if eval_epochs is None or eval_values is None:
            continue

        eval_epochs = np.asarray(eval_epochs)
        eval_values = np.asarray(eval_values, dtype=float)
        if len(eval_epochs) == 0 or len(eval_values) == 0:
            continue

        processed.append((eval_epochs, eval_values))
        for epoch, value in zip(eval_epochs, eval_values):
            epoch_to_values[int(epoch)].append(float(value))

    if not processed:
        return None, None

    mean_epochs = np.asarray(sorted(epoch_to_values.keys()))
    mean_values = np.asarray([np.mean(epoch_to_values[int(epoch)]) for epoch in mean_epochs], dtype=float)
    return processed, (mean_epochs, mean_values)

def compare_scaling(log_files, agent_counts, output_file='ic3net_scaling_comparisonPrviDan.png', smoothing_window=50):
    """
    Compare IC3Net performance across different agent counts with 4 key metrics
    
    Args:
        log_files: List of log file paths for different agent counts
        agent_counts: List of agent counts corresponding to each log file
        output_file: Output filename for the comparison plot
    """
    print("\n=== IC3Net Scaling Comparison ===")
    print(f"Comparing {len(log_files)} configurations...")
    
    # Load all logs
    all_data = []
    for log_file, nagents in zip(log_files, agent_counts):
        if not Path(log_file).exists():
            print(f"⚠ Warning: {log_file} not found, skipping...")
            continue
            
        try:
            print(f"Loading {log_file} ({nagents} agents)...")
            log_data = torch.load(log_file, weights_only=False)
            
            if isinstance(log_data, dict) and 'log' in log_data:
                log = log_data['log']
            else:
                log = log_data
            
            # Extract metrics
            data = {'nagents': nagents}
            
            if 'epoch' in log and hasattr(log['epoch'], 'data'):
                data['epochs'] = np.array(log['epoch'].data)
            
            if 'reward' in log and hasattr(log['reward'], 'data'):
                # Handle both scalar and array rewards
                rewards = []
                for r in log['reward'].data:
                    if isinstance(r, np.ndarray):
                        rewards.append(float(np.mean(r)))
                    else:
                        rewards.append(float(r))
                data['rewards'] = np.array(rewards)
            
            if 'success' in log and hasattr(log['success'], 'data'):
                data['success'] = np.array([float(s) for s in log['success'].data])
            
            if 'completion_rate' in log and hasattr(log['completion_rate'], 'data'):
                data['completion'] = np.array([float(c) for c in log['completion_rate'].data])
            
            if 'total_crashes' in log and hasattr(log['total_crashes'], 'data'):
                data['crashes'] = np.array([float(c) for c in log['total_crashes'].data])
            elif 'success' in data:
                # If crashes not available, estimate from success rate
                data['crashes'] = None
            
            if 'comm_action' in log and hasattr(log['comm_action'], 'data'):
                comm = []
                for c in log['comm_action'].data:
                    if isinstance(c, np.ndarray):
                        comm.append(float(np.mean(c)))
                    else:
                        comm.append(float(c))
                data['communication'] = np.array(comm)

            if 'eval_epoch' in log and hasattr(log['eval_epoch'], 'data'):
                data['eval_epochs'] = np.array([int(e) for e in log['eval_epoch'].data])
            if 'eval_reward' in log and hasattr(log['eval_reward'], 'data'):
                data['eval_rewards'] = _to_float_array(log['eval_reward'].data)
            if 'eval_success' in log and hasattr(log['eval_success'], 'data'):
                data['eval_success'] = _to_float_array(log['eval_success'].data)
            if 'eval_completion_rate' in log and hasattr(log['eval_completion_rate'], 'data'):
                data['eval_completion'] = _to_float_array(log['eval_completion_rate'].data)
            if 'eval_total_crashes' in log and hasattr(log['eval_total_crashes'], 'data'):
                data['eval_crashes'] = _to_float_array(log['eval_total_crashes'].data)
            
            all_data.append(data)
            print(f"  ✓ Loaded {len(data.get('epochs', []))} epochs")
            
        except Exception as e:
            print(f"  ✗ Error loading {log_file}: {e}")
            continue
    
    if len(all_data) == 0:
        print("No valid data to plot!")
        return
    
    grouped_data = defaultdict(list)
    for data in all_data:
        grouped_data[data['nagents']].append(data)

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IC3Net Scaling Analysis: Performance vs Agent Count', fontsize=16, fontweight='bold')
    
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    
    # Plot 1: Success Rate Over Training
    ax1 = axes[0, 0]
    for i, nagents in enumerate(sorted(grouped_data.keys())):
        processed, mean_data = _prepare_group_metric(grouped_data[nagents], 'success', smoothing_window)
        if not processed:
            continue

        color = colors[i % len(colors)]
        if len(processed) > 1:
            for epochs_smoothed, smoothed in processed:
                ax1.plot(epochs_smoothed, smoothed * 100, color=color, linewidth=1.1, alpha=0.22)
            label = f'{nagents} agents (mean of {len(processed)} seeds)'
        else:
            label = f'{nagents} agents'

        mean_epochs, mean_series = mean_data
        ax1.plot(mean_epochs, mean_series * 100, label=label, color=color, linewidth=3.0, alpha=1.0)

        eval_processed, eval_mean = _prepare_group_eval(grouped_data[nagents], 'eval_success')
        if eval_processed:
            if len(eval_processed) > 1:
                for eval_epochs, eval_values in eval_processed:
                    ax1.scatter(eval_epochs, eval_values * 100, color=color, s=18, alpha=0.25)
            mean_eval_epochs, mean_eval_values = eval_mean
            ax1.scatter(mean_eval_epochs, mean_eval_values * 100, color=color, s=48, alpha=1.0,
                        edgecolors='black', linewidths=0.5, zorder=5)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate (All Complete + No Crashes)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best')
    ax1.set_ylim([0, 105])
    
    # Plot 2: Total Crashes Per Episode
    ax2 = axes[0, 1]
    for i, nagents in enumerate(sorted(grouped_data.keys())):
        processed, mean_data = _prepare_group_metric(grouped_data[nagents], 'crashes', smoothing_window)
        if not processed:
            continue

        color = colors[i % len(colors)]
        if len(processed) > 1:
            for epochs_smoothed, smoothed in processed:
                ax2.plot(epochs_smoothed, smoothed, color=color, linewidth=1.1, alpha=0.22)
            label = f'{nagents} agents (mean of {len(processed)} seeds)'
        else:
            label = f'{nagents} agents'

        mean_epochs, mean_series = mean_data
        ax2.plot(mean_epochs, mean_series, label=label, color=color, linewidth=3.0, alpha=1.0)

        eval_processed, eval_mean = _prepare_group_eval(grouped_data[nagents], 'eval_crashes')
        if eval_processed:
            if len(eval_processed) > 1:
                for eval_epochs, eval_values in eval_processed:
                    ax2.scatter(eval_epochs, eval_values, color=color, s=18, alpha=0.25)
            mean_eval_epochs, mean_eval_values = eval_mean
            ax2.scatter(mean_eval_epochs, mean_eval_values, color=color, s=48, alpha=1.0,
                        edgecolors='black', linewidths=0.5, zorder=5)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Crashes', fontsize=12, fontweight='bold')
    ax2.set_title('Collision Events Per Episode', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best')
    
    # Plot 3: Average Reward Over Training
    ax3 = axes[1, 0]
    for i, nagents in enumerate(sorted(grouped_data.keys())):
        processed, mean_data = _prepare_group_metric(grouped_data[nagents], 'rewards', smoothing_window)
        if not processed:
            continue

        color = colors[i % len(colors)]
        if len(processed) > 1:
            for epochs_smoothed, smoothed in processed:
                ax3.plot(epochs_smoothed, smoothed, color=color, linewidth=1.1, alpha=0.22)
            label = f'{nagents} agents (mean of {len(processed)} seeds)'
        else:
            label = f'{nagents} agents'

        mean_epochs, mean_series = mean_data
        ax3.plot(mean_epochs, mean_series, label=label, color=color, linewidth=3.0, alpha=1.0)

        eval_processed, eval_mean = _prepare_group_eval(grouped_data[nagents], 'eval_rewards')
        if eval_processed:
            if len(eval_processed) > 1:
                for eval_epochs, eval_values in eval_processed:
                    ax3.scatter(eval_epochs, eval_values, color=color, s=18, alpha=0.25)
            mean_eval_epochs, mean_eval_values = eval_mean
            ax3.scatter(mean_eval_epochs, mean_eval_values, color=color, s=48, alpha=1.0,
                        edgecolors='black', linewidths=0.5, zorder=5)
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax3.set_title('Episode Reward (Higher = Better)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=11, loc='best')
    
    # Plot 4: Completion Rate
    ax4 = axes[1, 1]
    for i, nagents in enumerate(sorted(grouped_data.keys())):
        processed, mean_data = _prepare_group_metric(grouped_data[nagents], 'completion', smoothing_window)
        if not processed:
            continue

        color = colors[i % len(colors)]
        if len(processed) > 1:
            for epochs_smoothed, smoothed in processed:
                ax4.plot(epochs_smoothed, smoothed * 100, color=color, linewidth=1.1, alpha=0.22)
            label = f'{nagents} agents (mean of {len(processed)} seeds)'
        else:
            label = f'{nagents} agents'

        mean_epochs, mean_series = mean_data
        ax4.plot(mean_epochs, mean_series * 100, label=label, color=color, linewidth=3.0, alpha=1.0)

        eval_processed, eval_mean = _prepare_group_eval(grouped_data[nagents], 'eval_completion')
        if eval_processed:
            if len(eval_processed) > 1:
                for eval_epochs, eval_values in eval_processed:
                    ax4.scatter(eval_epochs, eval_values * 100, color=color, s=18, alpha=0.25)
            mean_eval_epochs, mean_eval_values = eval_mean
            ax4.scatter(mean_eval_epochs, mean_eval_values * 100, color=color, s=48, alpha=1.0,
                        edgecolors='black', linewidths=0.5, zorder=5)
    
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Agents Reaching Goal (Ignoring Crashes)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=11, loc='best')
    ax4.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_file}")
    plt.close()
    
    # Print summary statistics
    print("\n=== Training Summary ===")
    for data in all_data:
        nagents = data['nagents']
        print(f"\n{nagents} Agents:")
        if 'success' in data and len(data['success']) > 0:
            final_success = data['success'][-1] * 100
            avg_success_last50 = np.mean(data['success'][-50:]) * 100 if len(data['success']) >= 50 else final_success
            print(f"  Final Success Rate: {final_success:.1f}%")
            print(f"  Avg Success (last 50 epochs): {avg_success_last50:.1f}%")
        
        if 'crashes' in data and data['crashes'] is not None and len(data['crashes']) > 0:
            final_crashes = data['crashes'][-1]
            avg_crashes_last50 = np.mean(data['crashes'][-50:]) if len(data['crashes']) >= 50 else final_crashes
            print(f"  Final Crashes: {final_crashes:.1f}")
            print(f"  Avg Crashes (last 50 epochs): {avg_crashes_last50:.1f}")
        
        if 'rewards' in data and len(data['rewards']) > 0:
            final_reward = data['rewards'][-1]
            avg_reward_last50 = np.mean(data['rewards'][-50:]) if len(data['rewards']) >= 50 else final_reward
            print(f"  Final Reward: {final_reward:.2f}")
            print(f"  Avg Reward (last 50 epochs): {avg_reward_last50:.2f}")

def plot_training_log(log_file='run_log.pt'):
    """Plot training progress from saved log"""
    try:
        print(f"Loading log from {log_file}...")
        
        # Load the checkpoint - handle both old and new PyTorch versions
        try:
            log_data = torch.load(log_file, weights_only=False)
        except TypeError:
            # Older PyTorch version without weights_only parameter
            log_data = torch.load(log_file)
        
        # Extract log from the saved checkpoint
        if isinstance(log_data, dict) and 'log' in log_data:
            log = log_data['log']
        else:
            log = log_data
        
        print("Extracting metrics from log...")
        epochs = []
        rewards = []
        success_rates = []
        
        # Extract data from LogField objects
        if isinstance(log, dict):
            if 'epoch' in log and hasattr(log['epoch'], 'data'):
                epochs = log['epoch'].data
                print(f"  Found {len(epochs)} epochs")
            if 'reward' in log and hasattr(log['reward'], 'data'):
                rewards = log['reward'].data
                print(f"  Found {len(rewards)} reward entries")
            if 'success' in log and hasattr(log['success'], 'data'):
                success_rates = log['success'].data
                print(f"  Found {len(success_rates)} success entries")
        
        # Convert to numpy arrays, handling any nested structures
        if epochs:
            # Helper function to safely extract scalar values
            def to_scalar(val):
                if isinstance(val, (list, tuple)):
                    return float(val[0]) if len(val) > 0 else 0.0
                elif isinstance(val, np.ndarray):
                    if val.size == 1:
                        return float(val.item())
                    elif val.size > 0:
                        return float(val[0])
                    else:
                        return 0.0
                else:
                    return float(val)
            
            epochs = np.array([to_scalar(e) for e in epochs])
            rewards = np.array([to_scalar(r) for r in rewards])
            success_rates = np.array([to_scalar(s) for s in success_rates])
            
            print(f"Creating plot with {len(epochs)} data points...")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot rewards
            ax1.plot(epochs, rewards, 'b-', marker='o', markersize=3, label='Episode Reward')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Reward Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot success rate
            ax2.plot(epochs, success_rates, 'g-', marker='s', markersize=3, label='Success Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate Over Time')
            ax2.set_ylim([0, 1.1])
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            output_file = 'training_progress_mw.png'
            plt.savefig(output_file, dpi=100)
            print(f"✓ Saved {output_file}")
            plt.show()  # Display the plot
            plt.close()
            
            # Print summary statistics
            print("\n=== Training Summary ===")
            print(f"Total epochs: {len(epochs)}")
            print(f"Final reward: {rewards[-1]:.4f}")
            print(f"Final success rate: {success_rates[-1]:.4f}")
            if len(rewards) > 1:
                print(f"Average reward (last 10): {np.mean(rewards[-10:]):.4f}")
                print(f"Average success (last 10): {np.mean(success_rates[-10:]):.4f}")
        else:
            print("No training data found in log file")
            
    except FileNotFoundError:
        print(f"Log file '{log_file}' not found. Run training first.")
    except Exception as e:
        print(f"Error loading log file: {e}")
        
        traceback.print_exc()

def track_car_movement(num_episodes=5):
    """Track and visualize car movement in the environment"""
    try:
        from data import init
        
        # Create environment with same settings as training
        class Args:
            env_name = 'traffic_junction'
            nagents = 1
            dim = 6
            max_steps = 20
            difficulty = 'easy'
            vision = 1
        
        args = Args()
        env = init(args.env_name, args, final_init=False)
        
        fig, axes = plt.subplots(1, min(num_episodes, 5), figsize=(4*min(num_episodes, 5), 4))
        if num_episodes == 1:
            axes = [axes]
        
        print(f"Running {num_episodes} episodes to track car movement...")
        
        for ep in range(num_episodes):
            obs = env.reset()
            trajectory = []
            actions_taken = []
            
            for step in range(args.max_steps):
                # Take random action (for visualization)
                action = np.array([np.random.randint(0, 2)])
                obs, reward, done, info = env.step(action)
                actions_taken.append(action[0])
                
                # Extract car position
                unwrapped_env = env.env
                while hasattr(unwrapped_env, 'env'):
                    unwrapped_env = unwrapped_env.env
                
                if hasattr(unwrapped_env, 'car_loc'):
                    trajectory.append(unwrapped_env.car_loc.copy())
                
                if done:
                    break
            
            # Plot trajectory
            if ep < len(axes):
                ax = axes[ep]
                dims = (args.dim, args.dim)  # 6x6 grid
                
                ax.set_xlim(-0.5, dims[1] - 0.5)
                ax.set_ylim(-0.5, dims[0] - 0.5)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                
                # Draw grid
                for i in range(dims[0] + 1):
                    ax.axhline(y=i - 0.5, color='gray', alpha=0.2, linewidth=0.5)
                for j in range(dims[1] + 1):
                    ax.axvline(x=j - 0.5, color='gray', alpha=0.2, linewidth=0.5)
                
                # Draw trajectory
                if len(trajectory) > 0:
                    trajectory = np.array(trajectory)
                    path = trajectory[:, 0, :]  # First car
                    
                    # Plot path as line with gradient color
                    for i in range(len(path) - 1):
                        color_intensity = i / max(len(path) - 1, 1)
                        ax.plot(path[i:i+2, 1], path[i:i+2, 0], 'b-', 
                               alpha=0.3 + 0.7*color_intensity, linewidth=2)
                    
                    # Mark start and end
                    ax.plot(path[0, 1], path[0, 0], 'go', markersize=10, label='Start', zorder=5)
                    ax.plot(path[-1, 1], path[-1, 0], 'r*', markersize=15, label='End', zorder=5)
                    
                    # Add action labels
                    action_str = ''.join(['G' if a == 1 else 'B' for a in actions_taken[:5]])
                    ax.set_title(f'Episode {ep+1} - Actions: {action_str}...\n({len(path)} steps)')
                else:
                    ax.set_title(f'Episode {ep+1} - No movement')
                
                ax.set_xlabel('Column (X)')
                ax.set_ylabel('Row (Y)')
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('car_trajectories.png', dpi=100)
        print(f"✓ Saved car_trajectories.png")
        plt.show()
        plt.close()
        
        env.close()
        
        # Print interpretation
        print("\n=== Car Movement Analysis ===")
        print("Green circle: Starting position")
        print("Red star: Ending position")
        print("Blue gradient: Car path (darker=earlier, lighter=later)")
        print("G = Gas action, B = Brake action")
        
    except Exception as e:
        print(f"Error tracking car: {e}")
       
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize IC3Net training progress')
    parser.add_argument('--plot-training', action='store_true', help='Plot training progress')
    parser.add_argument('--compare-scaling', action='store_true', help='Compare multiple agent counts (4-plot comparison)')
    parser.add_argument('--track-car', action='store_true', help='Track car movement in environment')
    parser.add_argument('--log_file', default='run_log.pt', type=str, help='Path to the log file')
    parser.add_argument('--log_files', nargs='+', help='List of log files for comparison (e.g., run_log_5.pt run_log_10.pt run_log_20.pt)')
    parser.add_argument('--agent_counts', nargs='+', type=int, help='Agent counts for each log file (e.g., 5 10 20)')
    parser.add_argument('--num_episodes', default=5, type=int, help='Number of episodes to track')
    parser.add_argument('--output', default='ic3net_scaling_comparison.png', type=str, help='Output filename for comparison plot')
    parser.add_argument('--smooth', default=50, type=int, help='Smoothing window for plots (epochs)')
    args = parser.parse_args()
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if args.compare_scaling:
        print("\n=== Scaling Comparison Mode ===")
        
        # Auto-detect log files if not specified
        if args.log_files is None:
            # Try to find common log files
            possible_files = [
                ('run_log_5.pt', 5),
                ('run_log_5agents.pt', 5),
                ('run_log_10.pt', 10),
                ('run_log_10agents.pt', 10),
                ('run_log_20.pt', 20),
                ('run_log_20agents.pt', 20),
            ]
            
            found_files = []
            found_counts = []
            for fname, count in possible_files:
                if Path(fname).exists() and count not in found_counts:
                    found_files.append(fname)
                    found_counts.append(count)
            
            if len(found_files) == 0:
                print("No log files found! Please specify with --log_files and --agent_counts")
                print("Example: python visualize_training.py --compare-scaling --log_files run_log_5.pt run_log_10.pt --agent_counts 5 10")
                sys.exit(1)
            
            args.log_files = found_files
            args.agent_counts = found_counts
            print(f"Auto-detected {len(found_files)} log files: {found_files}")
        
        if args.agent_counts is None:
            print("Error: --agent_counts required when using --log_files")
            print("Example: --log_files run_log_5.pt run_log_10.pt --agent_counts 5 10")
            sys.exit(1)
        
        if len(args.log_files) != len(args.agent_counts):
            print(f"Error: Number of log files ({len(args.log_files)}) must match number of agent counts ({len(args.agent_counts)})")
            sys.exit(1)
        
        compare_scaling(args.log_files, args.agent_counts, args.output, args.smooth)
        
    elif args.track_car:
        print("\n=== Car Tracking Mode ===")
        print("Tracking car movement across episodes...")
        try:
            from data import init
            from utils import *
            track_car_movement(args.num_episodes)
        except Exception as e:
            print(f"Error in car tracking: {e}")
            import traceback
            traceback.print_exc()
    elif args.plot_training or not any([args.compare_scaling, args.track_car]):
        # Default behavior: plot training if no args or if --plot-training is specified
        plot_training_log(args.log_file)
    else:
        print("Usage:")
        print("  python visualize_training.py --plot-training           # Plot single training run")
        print("  python visualize_training.py --compare-scaling         # Compare multiple agent counts (auto-detect)")
        print("  python visualize_training.py --compare-scaling --log_files run_log_5.pt run_log_10.pt --agent_counts 5 10")
        print("  python visualize_training.py --track-car --num_episodes N  # Track car in N episodes")

