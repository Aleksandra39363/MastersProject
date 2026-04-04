#!/usr/bin/env python
"""
Real-time grid visualization of car navigation - SIMPLIFIED VERSION
Shows the actual grid with car moving step-by-step using trained model
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import sys
import argparse
from data import init
from comm import CommNetMLP
from action_utils import parse_action_args
from utils import init_args_for_env

def visualize_car_real_time(num_episodes=3, model_path='model.pt', delay=0.5):
    """Visualize car navigation with trained policy"""
    
    # Create parser and args exactly like main.py does
    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument('--env_name', default="traffic_junction")
    parser.add_argument('--nagents', type=int, default=1)
    parser.add_argument('--max_steps', default=20, type=int)
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--recurrent', action='store_true', default=False)
    parser.add_argument('--commnet', action='store_true', default=False)
    parser.add_argument('--ic3net', action='store_true', default=False)
    parser.add_argument('--hard_attn', default=False, action='store_true')
    parser.add_argument('--comm_mode', type=str, default='avg')
    parser.add_argument('--comm_passes', type=int, default=1)
    parser.add_argument('--comm_mask_zero', action='store_true', default=False)
    parser.add_argument('--mean_ratio', default=1.0, type=float)
    parser.add_argument('--rnn_type', default='MLP', type=str)
    parser.add_argument('--detach_gap', default=10000, type=int)
    parser.add_argument('--comm_init', default='uniform', type=str)
    parser.add_argument('--comm_action_one', default=False, action='store_true')
    parser.add_argument('--advantages_per_action', default=False, action='store_true')
    parser.add_argument('--share_weights', default=False, action='store_true')
    parser.add_argument('--nactions', default='1', type=str)
    parser.add_argument('--batch_size', type=int, default=500)
    
    # Initialize environment-specific args
    init_args_for_env(parser)
    
    # Parse arguments (without hardcoded values - use defaults from parser)
    args = parser.parse_args([
        '--ic3net',
        '--nagents', '2',  # Match training configuration
        '--max_steps', '40',  # Match training configuration
        '--hid_size', '128'  # Match training configuration
    ])
    
    # Set IC3Net flags (copied from main.py)
    if args.ic3net:
        args.commnet = 1
        args.hard_attn = 1
        args.mean_ratio = 0
        args.comm_action_one = True
    
    args.nfriendly = args.nagents
    args.display = False  # Don't use curses display (doesn't work well on Windows)
    
    # Ensure required environment attributes exist (with defaults matching training)
    if not hasattr(args, 'dim'):
        args.dim = 6
    if not hasattr(args, 'vision'):
        args.vision = 2
    if not hasattr(args, 'difficulty'):
        args.difficulty = 'easy'
    if not hasattr(args, 'add_rate_min'):
        args.add_rate_min = 1.0  # Match training: guaranteed spawn
    if not hasattr(args, 'add_rate_max'):
        args.add_rate_max = 1.0  # Match training: guaranteed spawn
    if not hasattr(args, 'curr_start'):
        args.curr_start = 0
    if not hasattr(args, 'curr_end'):
        args.curr_end = 0
    if not hasattr(args, 'vocab_type'):
        args.vocab_type = 'bool'
    
    # Initialize environment
    env = init(args.env_name, args, False)
    
    num_inputs = env.observation_dim
    args.num_actions = env.num_actions
    
    if not isinstance(args.num_actions, (list, tuple)):
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs
    
    # Hard attention adds communication action
    if args.hard_attn and args.commnet:
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1
    
    # Recurrence
    if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
        args.recurrent = True
        args.rnn_type = 'LSTM'
    
    parse_action_args(args)
    
    print(f"✓ Args configured: nagents={args.nagents}, dim_actions={args.dim_actions}, num_actions={args.num_actions}")
    
    # Load trained policy
    policy_net = None
    try:
        policy_net = CommNetMLP(args, num_inputs)
        # Try loading with weights_only parameter for newer PyTorch, fallback for older versions
        try:
            checkpoint = torch.load(model_path, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path)
        
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net'])
        else:
            policy_net.load_state_dict(checkpoint)
        
        # Convert entire model to float32 AFTER loading state dict
        policy_net = policy_net.float()
        policy_net.eval()
        print(f"✓ Loaded trained policy from {model_path}")
        print(f"  Model architecture: nagents={args.nagents}, inputs={num_inputs}, actions={args.num_actions}\n")
    except Exception as e:
        print(f"⚠ Could not load model from {model_path}")
        print(f"  Error: {e}")
        print(f"  Using RANDOM policy for demo")
        print(f"⚠ Random policy was used (model not loaded). Train first with:")
        print(f"   python main.py --nagents {args.nagents} --ic3net --save model.pt\n")
        policy_net = None
    
    # Get unwrapped environment
    unwrapped_env = env.env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    print("="*70)
    print("REAL-TIME CAR NAVIGATION VISUALIZATION")
    print("="*70)
    print(f"Policy: {'✓ TRAINED (IC3Net)' if policy_net else '✗ RANDOM (model not loaded!)'}")
    print(f"Grid: {args.dim}×{args.dim}")
    print(f"Max steps: {args.max_steps}")
    print(f"Episodes: {num_episodes}\n")
    
    all_successes = []
    all_rewards = []
    all_steps = []
    
    for ep in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep+1}/{num_episodes}")
        print('='*70)
        
        obs = env.reset()
        
        # Get actual grid dimensions (easy mode is dim+1)
        actual_dim = unwrapped_env.dims[0] if hasattr(unwrapped_env, 'dims') else args.dim
        
        # Route will be set after first step when car spawns
        print(f"Grid dimensions: {actual_dim}x{actual_dim}")
        print(f"Add rate: {unwrapped_env.add_rate if hasattr(unwrapped_env, 'add_rate') else 'unknown'}")
        print(f"Waiting for car to spawn...")
        
        start_cell, end_cell = None, None
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        plt.show()
        
        episode_reward = 0
        hidden_state = None
        success = False
        
        # Initialize info dict with comm_action (required for IC3Net)
        info = {'comm_action': np.zeros(args.nagents, dtype=int)}
        
        for step in range(args.max_steps):
            # Get action for ALL agents
            if policy_net:
                with torch.no_grad():
                    # Convert observation to float32 tensor
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.float()
                    else:
                        obs_tensor = torch.from_numpy(np.array(obs)).float()
                    
                    if obs_tensor.dim() == 2:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    if hidden_state is None:
                        output, hidden_state = policy_net(obs_tensor, info=info)
                    else:
                        output, hidden_state = policy_net(obs_tensor, hidden_state, info=info)
                    
                    # Get movement action (first action dimension) for ALL agents
                    action_logits = output[0][0, :, :]  # First action dim, shape: (nagents, num_actions)
                    action = torch.argmax(action_logits, dim=-1).cpu().numpy()
                    
                    # Get communication action if IC3Net (second action dimension)
                    if args.hard_attn and len(output) > 1:
                        comm_action_logits = output[1][0, :, :]  # Second action dim
                        comm_action = torch.argmax(comm_action_logits, dim=-1).cpu().numpy()
                        info['comm_action'] = comm_action
                    
                    # Ensure action has shape (nagents,)
                    if action.ndim == 0:
                        action = np.array([action.item()])
                    elif action.ndim > 1:
                        action = action.flatten()
            else:
                # Random action for each agent
                action = np.array([np.random.randint(0, 2) for _ in range(args.nagents)])
            
            # Step
            obs, reward, done, info = env.step(action)
            
            # Handle reward (can be scalar or array for multi-agent)
            if isinstance(reward, (np.ndarray, list)):
                reward_scalar = float(np.sum(reward))  # Sum rewards for all agents
            else:
                reward_scalar = float(reward)
            episode_reward += reward_scalar
            
            # Get ALL car positions and alive status
            all_car_positions = []
            if hasattr(unwrapped_env, 'car_loc') and hasattr(unwrapped_env, 'alive_mask'):
                for idx in range(args.nagents):
                    if unwrapped_env.alive_mask[idx] == 1:
                        car_pos = unwrapped_env.car_loc[idx]
                        all_car_positions.append((idx, car_pos))
            
            # Check success
            if hasattr(unwrapped_env, 'stat') and 'success' in unwrapped_env.stat:
                success = unwrapped_env.stat['success'] == 1
            
            # Update route info if first car just spawned
            if len(all_car_positions) > 0 and (start_cell is None) and hasattr(unwrapped_env, 'chosen_path'):
                if len(unwrapped_env.chosen_path) > 0 and len(unwrapped_env.chosen_path[0]) > 0:
                    route = unwrapped_env.chosen_path[0]
                    start_cell = tuple(route[0])
                    end_cell = tuple(route[-1])
                    print(f"\nCar spawned! Route: {start_cell} → {end_cell}")
                    print(f"Distance: {abs(end_cell[0]-start_cell[0]) + abs(end_cell[1]-start_cell[1])} cells")
            
            # Draw
            ax.clear()
            ax.set_xlim(-0.5, actual_dim - 0.5)
            ax.set_ylim(-1.2, actual_dim - 0.5)  # Minimal space
            ax.set_aspect('equal')
            ax.set_title(f'Traffic Junction - Step {step+1}/{args.max_steps}', 
                        fontsize=10, weight='bold', pad=5)
            ax.set_xticks(range(actual_dim))
            ax.set_yticks(range(actual_dim))
            ax.invert_yaxis()  # Invert y-axis so (0,0) is at top-left like array indexing
            
            # Draw background (grass/outside area)
            for i in range(actual_dim):
                for j in range(actual_dim):
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, 
                                   facecolor='lightgreen', 
                                   edgecolor='darkgreen', linewidth=0.5, alpha=0.3)
                    ax.add_patch(rect)
            
            # Draw complete crossroads (vertical and horizontal roads)
            mid = actual_dim // 2
            # Vertical road (TOP to BOTTOM)
            for i in range(actual_dim):
                rect = Rectangle((mid-0.5, i-0.5), 1, 1, 
                               facecolor='gray', edgecolor='white', linewidth=1)
                ax.add_patch(rect)
            # Horizontal road (LEFT to RIGHT)
            for j in range(actual_dim):
                rect = Rectangle((j-0.5, mid-0.5), 1, 1, 
                               facecolor='gray', edgecolor='white', linewidth=1)
                ax.add_patch(rect)
            
            # Draw junction center (intersection) with special highlight
            junction = Rectangle((mid-0.5, mid-0.5), 1, 1, 
                               facecolor='yellow', edgecolor='orange', 
                               linewidth=2, alpha=0.5)
            ax.add_patch(junction)
            
            # Draw road markings (dashed center lines)
            # Vertical road center line
            for i in range(actual_dim):
                if i != mid:  # Skip junction
                    ax.plot([mid, mid], [i-0.3, i+0.3], 'w--', linewidth=1.5, alpha=0.7)
            # Horizontal road center line  
            for j in range(actual_dim):
                if j != mid:  # Skip junction
                    ax.plot([j-0.3, j+0.3], [mid, mid], 'w--', linewidth=1.5, alpha=0.7)
            
            # Draw complete route path if available
            if len(all_car_positions) > 0 and hasattr(unwrapped_env, 'chosen_path') and len(unwrapped_env.chosen_path) > 0:
                if len(unwrapped_env.chosen_path[0]) > 0:
                    route_path = unwrapped_env.chosen_path[0]
                    for i in range(len(route_path) - 1):
                        p1 = route_path[i]
                        p2 = route_path[i + 1]
                        # Draw as line on the grid (using j,i since x is column, y is row)
                        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 
                               'cyan', alpha=0.4, linewidth=4, linestyle='-', zorder=2)
            
            # Draw destination (as red star with FINISH label)
            if end_cell is not None:
                try:
                    if isinstance(end_cell, np.ndarray):
                        dest_coords = end_cell.flatten()
                        dest_row, dest_col = float(dest_coords[0]), float(dest_coords[1])
                    elif isinstance(end_cell, (tuple, list)):
                        dest_row, dest_col = float(end_cell[0]), float(end_cell[1])
                    else:
                        dest_row = dest_col = float(end_cell)
                    
                    # Draw larger red star for destination
                    ax.plot(dest_col, dest_row, 'r*', markersize=40, zorder=5, 
                           markeredgewidth=2.5, markeredgecolor='darkred')
                    # Add FINISH label below destination
                    ax.text(dest_col, dest_row + 0.4, 'FINISH', ha='center', 
                           fontsize=7, weight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                   edgecolor='red', linewidth=1, alpha=0.9))
                    
                    # Show START position if we have route info
                    if start_cell is not None and start_cell != end_cell:
                        start_row, start_col = start_cell[0], start_cell[1]
                        if start_row != dest_row or start_col != dest_col:
                            ax.plot(start_col, start_row, 'g^', markersize=25, zorder=5,
                                   markeredgewidth=2, markeredgecolor='darkgreen')
                            ax.text(start_col, start_row - 0.4, 'START', ha='center',
                                   fontsize=6, weight='bold', color='green',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor='lightgreen',
                                           edgecolor='darkgreen', linewidth=0.8, alpha=0.9))
                except Exception as e:
                    pass
            
            # Draw ALL cars (positions are row,col; plot as col,row with inverted y-axis)
            for car_idx, car_pos in all_car_positions:
                car_row, car_col = car_pos[0], car_pos[1]
                
                # Color: green if success, blue otherwise
                if success:
                    color = 'green'
                    edge_color = 'darkgreen'
                else:
                    color = 'blue'
                    edge_color = 'darkblue'
                
                # Draw car as circle with border
                circle = Circle((car_col, car_row), 0.35, 
                              facecolor=color, edgecolor=edge_color, 
                              linewidth=2.5, zorder=10)
                ax.add_patch(circle)
                
                # Add car label with index
                ax.text(car_col, car_row, str(car_idx), 
                       ha='center', va='center', fontsize=7, weight='bold', 
                       color='white', zorder=11)
            
            # Show completion/crash status banner (compact)
            any_alive = len(all_car_positions) > 0
            if success:
                ax.text(actual_dim/2, -1.0, 'DONE', 
                       ha='center', fontsize=8, weight='bold', color='green',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', 
                               edgecolor='darkgreen', linewidth=1.5, alpha=0.95))
            elif not any_alive and step > 0:
                ax.text(actual_dim/2, -1.0, 'CRASHED', 
                       ha='center', fontsize=8, weight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                               edgecolor='red', linewidth=1.5, alpha=0.95))
            
            # Add compact info box
            # Show actions for alive cars
            action_parts = []
            for car_idx, car_pos in all_car_positions:
                if car_idx < len(action):
                    act_text = 'GAS' if action[car_idx] == 1 else 'BRAKE'
                    action_parts.append(f"Car{car_idx}:{act_text}")
            
            action_text = " | ".join(action_parts) if action_parts else "No cars"
            info_text = f"Step {step+1}/{args.max_steps} | {action_text}"
            
            ax.text(actual_dim/2, -0.6, info_text, ha='center',
                   fontsize=7, verticalalignment='top', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', 
                           edgecolor='darkblue', linewidth=1, alpha=0.9))
            
            plt.draw()
            plt.pause(delay)
            
            if done:
                plt.pause(1.5)  # Show final state longer
                break
        
        plt.close(fig)
        
        # Stats
        all_successes.append(success)
        all_rewards.append(episode_reward)
        all_steps.append(step + 1)
        
        status = '✓ SUCCESS' if success else '✗ FAILED'
        print(f"\n{status}")
        print(f"  Steps taken: {step+1}/{args.max_steps}")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Final position: {car_pos}")
        if end_cell:
            print(f"  Target was: {end_cell}")
    
    # Final stats
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print('='*70)
    print(f"Success rate: {np.mean(all_successes)*100:.1f}%")
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Average reward: {np.mean(all_rewards):.4f}")
    print('='*70)
    
    if policy_net and np.mean(all_successes) == 1.0:
        print("\n🎉 Perfect! Trained policy reaches destination 100% of the time!")
    elif policy_net and np.mean(all_successes) > 0:
        print(f"\n✓ Trained policy working! Success rate: {np.mean(all_successes)*100:.0f}%")
    elif not policy_net:
        print("\n⚠ Random policy was used (model not loaded). Train first with:")
        print("   python main.py --nagents 1 --ic3net --save model.pt")
    else:
        print("\n✗ Policy not reaching destination. May need more training.")
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--model', type=str, default='model.pt')
    parser.add_argument('--delay', type=float, default=0.5)
    args = parser.parse_args()
    
    visualize_car_real_time(args.episodes, args.model, args.delay)
