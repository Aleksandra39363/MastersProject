#!/usr/bin/env python
"""
Real-time grid visualization of car navigation
Shows the actual grid with car moving step-by-step
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import time
from data import init
from comm import CommNetMLP

def visualize_car_navigation(num_episodes=5, model_path='model.pt', delay=0.3):
    """
    Visualize car moving in real-time on the grid
    
    Args:
        num_episodes: Number of episodes to run
        model_path: Path to trained model
        delay: Delay between steps (seconds)
    """
    
    # Load model first to get args from checkpoint
    policy_net = None
    args = None
    
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        if isinstance(checkpoint, dict) and 'args' in checkpoint:
            # Args saved in checkpoint
            args = checkpoint['args']
            print(f"✓ Loaded args from {model_path}")
        else:
            print(f"⚠ Args not found in checkpoint, using defaults")
    except Exception as e:
        print(f"⚠ Could not load checkpoint: {e}")
    
    # If no args from checkpoint, create minimal args
    if args is None:
        class Args:
            env_name = 'traffic_junction'
            nagents = 1
            dim = 6
            max_steps = 20
            difficulty = 'easy'
            vision = 1
            display = False
            ncar = 1
            ntraffic = 0
            enemy_comm = False
            shared_reward = False
            add_rate_min = 0.05
            add_rate_max = 0.2
            curr_start = 0
            curr_end = 0
            vocab_type = 'bool'
        args = Args()
    
    env = init(args.env_name, args, final_init=False)
    
    # Now try to load the policy network with proper args
    if args and checkpoint:
        try:
            from comm import CommNetMLP
            num_inputs = env.observation_dim
            policy_net = CommNetMLP(args, num_inputs)
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                policy_net.load_state_dict(checkpoint['policy_net'])
            else:
                policy_net.load_state_dict(checkpoint)
            policy_net.eval()
            print(f"✓ Loaded trained policy from {model_path}\n")
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("  Using random policy\n")
            policy_net = None
    else:
        print("  Using random policy\n")
    
    # Get unwrapped environment for direct access
    unwrapped_env = env.env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    
    # Statistics
    all_successes = []
    all_rewards = []
    all_steps = []
    
    print("="*70)
    print("REAL-TIME CAR NAVIGATION VISUALIZATION")
    print("="*70)
    print(f"Policy: {'TRAINED (IC3Net)' if policy_net else 'RANDOM'}")
    print(f"Grid: {args.dim}×{args.dim}")
    print(f"Max steps: {args.max_steps}")
    print(f"Episodes: {num_episodes}\n")
    
    for ep in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep+1}/{num_episodes}")
        print('='*70)
        
        obs = env.reset()
        
        # Get route info
        route_id = unwrapped_env.route_id[0] if hasattr(unwrapped_env, 'route_id') else 0
        routes = unwrapped_env.routes if hasattr(unwrapped_env, 'routes') else []
        
        if route_id < len(routes):
            route = routes[route_id]
            # Route is array of [x,y] positions
            start_cell = tuple(route[0]) if len(route) > 0 else None
            end_cell = tuple(route[-1]) if len(route) > 0 else None
            print(f"Route: {start_cell} → {end_cell}")
        else:
            start_cell = None
            end_cell = None
            print("Route: Unknown")
        
        # Initialize plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        plt.show()
        
        episode_reward = 0
        hidden_state = None if policy_net else None
        
        for step in range(args.max_steps):
            # Get action from policy
            if policy_net:
                with torch.no_grad():
                    obs_tensor = obs.float() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs).float()
                    if obs_tensor.dim() == 2:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    if hidden_state is None:
                        output, hidden_state = policy_net(obs_tensor)
                    else:
                        output, hidden_state = policy_net(obs_tensor, hidden_state)
                    
                    # Get action (movement only, ignore communication)
                    action_logits = output[0]  # (dim_actions, nagents, num_actions)
                    action = torch.argmax(action_logits[0, :, :], dim=-1).cpu().numpy()  # First action dim
                    action = action.flatten()
            else:
                # Random policy
                action = np.array([np.random.randint(0, 2)])
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            reward_scalar = float(reward) if isinstance(reward, (np.ndarray, list)) else reward
            episode_reward += reward_scalar
            
            # Get current state
            car_pos = unwrapped_env.car_loc[0] if hasattr(unwrapped_env, 'car_loc') else [0, 0]
            alive = unwrapped_env.alive_mask[0] if hasattr(unwrapped_env, 'alive_mask') else 1
            
            # Check if reached destination
            success = False
            if hasattr(unwrapped_env, 'stat') and 'success' in unwrapped_env.stat:
                success = unwrapped_env.stat['success'] == 1
            elif 'is_completed' in info and len(info['is_completed']) > 0:
                success = info['is_completed'][0] == 1
            
            # Draw grid
            ax.clear()
            ax.set_xlim(-0.5, args.dim - 0.5)
            ax.set_ylim(-0.5, args.dim - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Episode {ep+1} - Step {step+1}/{args.max_steps} - Reward: {episode_reward:.2f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position', fontsize=12)
            ax.set_ylabel('Y Position', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(args.dim))
            ax.set_yticks(range(args.dim))
            
            # Draw grid cells
            for i in range(args.dim):
                for j in range(args.dim):
                    rect = Rectangle((i-0.5, j-0.5), 1, 1, 
                                    facecolor='lightgray', 
                                    edgecolor='black', 
                                    linewidth=1)
                    ax.add_patch(rect)
            
            # Draw destination
            if end_cell is not None:
                # end_cell is tuple (x, y)
                dest_x, dest_y = end_cell[0], end_cell[1]
                dest_rect = Rectangle((dest_x-0.5, dest_y-0.5), 1, 1,
                                      facecolor='lightgreen', 
                                      edgecolor='green', 
                                      linewidth=3,
                                      alpha=0.5)
                ax.add_patch(dest_rect)
                ax.text(dest_x, dest_y, '★\nGOAL', 
                       ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='darkgreen')
            
            # Draw car
            if alive:
                car_circle = Circle((car_pos[0], car_pos[1]), 0.35,
                                   facecolor='red' if not success else 'blue',
                                   edgecolor='darkred' if not success else 'darkblue',
                                   linewidth=2)
                ax.add_patch(car_circle)
                ax.text(car_pos[0], car_pos[1], '🚗', 
                       ha='center', va='center', fontsize=24)
            
            # Action text
            action_text = 'GAS' if action[0] == 1 else 'BRAKE'
            ax.text(args.dim//2, -1.5, f'Action: {action_text}', 
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Status text
            if success:
                status_text = '✓ REACHED DESTINATION!'
                status_color = 'green'
            elif not alive:
                status_text = '✗ CRASHED'
                status_color = 'red'
            else:
                status_text = '● NAVIGATING...'
                status_color = 'blue'
            
            ax.text(args.dim//2, args.dim + 0.5, status_text,
                   ha='center', fontsize=14, fontweight='bold',
                   color=status_color)
            
            plt.draw()
            plt.pause(delay)
            
            if done:
                # Show final state a bit longer
                plt.pause(1.0)
                break
        
        plt.close(fig)
        
        # Episode summary
        all_successes.append(success)
        all_rewards.append(episode_reward)
        all_steps.append(step + 1)
        
        status = '✓ SUCCESS' if success else '✗ TIMEOUT/CRASH'
        print(f"\n{status}")
        print(f"  Steps: {step+1}")
        print(f"  Reward: {episode_reward:.4f}")
        print(f"  Final position: {car_pos}")
    
    # Final statistics
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print('='*70)
    print(f"Success rate: {np.mean(all_successes)*100:.1f}%")
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Average reward: {np.mean(all_rewards):.4f}")
    print('='*70)
    
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--model', type=str, default='model.pt', help='Model path')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between steps (seconds)')
    args = parser.parse_args()
    
    visualize_car_navigation(args.episodes, args.model, args.delay)
