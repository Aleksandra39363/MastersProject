#!/usr/bin/env python
"""
Track and visualize car movement using trained policy
Shows car positions, paths, and actions taken with the learned IC3Net model
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import sys
import argparse as argparse_module
from data import init
from comm import CommNetMLP
from action_utils import parse_action_args
from utils import init_args_for_env

# Set default tensor type to Double (like main.py does)
try:
    torch.set_default_dtype(torch.float64)
except Exception:
    try:
        torch.set_default_tensor_type('torch.DoubleTensor')
    except Exception:
        pass

def track_car_with_policy(num_episodes=5, model_path=None, render=True):
        
        # Create parser and args EXACTLY like main.py
        parser = argparse_module.ArgumentParser()
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
        parser.add_argument('--display', action='store_true', default=False)
        
        # Must call init_args_for_env with env_name already in sys.argv
        # Temporarily modify sys.argv so init_args_for_env can detect env_name
        old_argv = sys.argv
        sys.argv = ['track_car.py', '--env_name', 'traffic_junction']
        init_args_for_env(parser)
        sys.argv = old_argv
        
        args = parser.parse_args(['--ic3net', '--nagents', '1', '--env_name', 'traffic_junction', '--dim', '6', '--hid_size', '128'])
        
        # Apply IC3Net settings (from main.py)
        if args.ic3net:
            args.commnet = 1
            args.hard_attn = 1
            args.mean_ratio = 0
            args.comm_action_one = True
        
        args.nfriendly = args.nagents
        
        env = init(args.env_name, args, False)
        
        num_inputs = env.observation_dim
        args.num_actions = env.num_actions
        
        if not isinstance(args.num_actions, (list, tuple)):
            args.num_actions = [args.num_actions]
        args.dim_actions = env.dim_actions
        args.num_inputs = num_inputs
        
        # Hard attention adds comm action
        if args.hard_attn and args.commnet:
            args.num_actions = [*args.num_actions, 2]
            args.dim_actions = env.dim_actions + 1
        
        # Recurrence
        if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
            args.recurrent = True
            args.rnn_type = 'LSTM'
        
        parse_action_args(args)
        parse_action_args(args)
        
        print(f"✓ Args initialized: continuous={args.continuous}, naction_heads={args.naction_heads}")
        print(f"  num_actions={args.num_actions}, dim_actions={args.dim_actions}")
        
        # Load trained policy
        if model_path is None:
            model_path = 'model.pt'
        
        policy_net = None
        try:
            policy_net = CommNetMLP(args, num_inputs)
            # Try with weights_only first (newer PyTorch), fallback to without (older PyTorch)
            try:
                checkpoint = torch.load(model_path, weights_only=False)
            except TypeError:
                checkpoint = torch.load(model_path)
            
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                # Use strict=False to allow partial loading (architecture may have changed)
                policy_net.load_state_dict(checkpoint['policy_net'], strict=False)
            else:
                policy_net.load_state_dict(checkpoint, strict=False)
            policy_net.eval()
            print(f"✓ Loaded trained policy from {model_path} (some weights may be missing/extra)")
        except FileNotFoundError:
            print(f"⚠ Model not found at {model_path}")
            print("  Using random policy for demonstration")
            policy_net = None
        except Exception as e:
            print(f"⚠ Could not load model: {e}")
            print("  Using random policy for demonstration")
            policy_net = None
            import traceback
            traceback.print_exc()
        
        # Get environment info
        unwrapped_env = env.env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        
        print("\n" + "="*70)
        print("CAR TRACKING WITH TRAINED POLICY")
        print("="*70)
        print(f"Environment: Traffic Junction (6×6 grid)")
        print(f"Model: IC3Net (CommNetMLP + LSTM)")
        print(f"Episodes: {num_episodes}")
        print(f"Policy mode: {'TRAINED' if policy_net else 'RANDOM (for demo)'}\n")
        
        all_trajectories = []
        all_actions = []
        all_rewards = []
        all_successes = []
        all_crashes = []
        all_steps = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            trajectory = []
            actions_taken = []
            rewards_taken = []
            episode_reward = 0
            terminated = False
            
            # Get starting position
            if hasattr(unwrapped_env, 'car_loc'):
                start_pos = unwrapped_env.car_loc.copy()[0]
            else:
                start_pos = None
            
            for step in range(args.max_steps):
                if policy_net:
                    # Use trained policy
                    with torch.no_grad():
                        # Ensure double precision (model was trained with DoubleTensor)
                        obs_tensor = obs.double() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs).double()
                        if obs_tensor.dim() == 2:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        
                        # Create info dict for comm_action (required by IC3Net)
                        info_dict = {'comm_action': np.ones(args.nagents)}
                        
                        # Call model - it manages hidden state internally
                        output = policy_net(obs_tensor, info_dict)
                        
                        # Output is (action_list, value, hidden_state_tuple) 
                        # action_list is a list of log probabilities for each action dimension
                        action_logits = output[0]  # List of tensors
                        
                        # Get first action dimension (movement: brake=0, gas=1)
                        # action_logits[0] shape: (batch, nagents, num_actions)
                        log_p_a = action_logits[0]  # First action dimension
                        
                        # Sample from probability distribution
                        probs = torch.exp(log_p_a[0, 0, :])  # batch=0, agent=0, all actions
                        action_sample = torch.multinomial(probs, 1).item()
                        action = np.array([action_sample])
                else:
                    # Random action for demo
                    action = np.array([np.random.randint(0, 2)])
                
                actions_taken.append(action[0])
                obs, reward, done, info = env.step(action)
                
                reward_scalar = float(reward) if isinstance(reward, (np.ndarray, list)) else reward
                episode_reward += reward_scalar
                rewards_taken.append(reward_scalar)
                
                # Extract car position
                if hasattr(unwrapped_env, 'car_loc'):
                    pos = unwrapped_env.car_loc.copy()[0]
                    trajectory.append(pos)
                
                # Check for success (reached destination) and crash status
                success = False
                crashed = False
                
                # Check if completed successfully
                if 'is_completed' in info and len(info['is_completed']) > 0:
                    success = info['is_completed'][0] == 1
                elif hasattr(unwrapped_env, 'is_completed') and len(unwrapped_env.is_completed) > 0:
                    success = unwrapped_env.is_completed[0] == 1
                
                # Check if crashed
                if hasattr(unwrapped_env, 'has_failed'):
                    crashed = unwrapped_env.has_failed == 1
                
                # Only mark as success if completed AND not crashed
                if crashed:
                    success = False
                
                if done:
                    terminated = success  # Only count as success if actually reached destination without crash
                    break
            
            all_trajectories.append(trajectory)
            all_actions.append(actions_taken)
            all_rewards.append(rewards_taken)
            all_successes.append(terminated)
            all_crashes.append(crashed)
            all_steps.append(len(trajectory))
            
            # Print episode summary
            success_marker = "✓" if terminated else ("✗ CRASH" if crashed else "✗ TIMEOUT")
            print(f"Episode {ep+1}: {success_marker}")
            print(f"  Start position: {start_pos}")
            if len(trajectory) > 0:
                print(f"  End position:   {trajectory[-1]}")
                print(f"  Path length: {len(trajectory)} steps")
                if len(trajectory) > 1:
                    dist = np.linalg.norm(trajectory[-1] - start_pos)
                    print(f"  Distance from start: {dist:.2f} cells")
            # Count actions correctly: action 0=GAS (move), action 1=BRAKE (wait)
            gas_count = sum(1 for a in actions_taken if a == 0)
            brake_count = sum(1 for a in actions_taken if a == 1)
            print(f"  Actions: {' '.join(['G' if a == 0 else 'B' for a in actions_taken[:20]])}")
            print(f"  Action summary: {gas_count} GAS (move), {brake_count} BRAKE (wait)")
            print(f"  Reward: {episode_reward:.4f}")
            print(f"  Success: {terminated}")
            print(f"  Note: BRAKE actions cause car to wait in place (appears frozen)\\n")
        
        # Statistics
        print("="*70)
        print("EPISODE STATISTICS")
        print("="*70)
        print(f"Success rate: {np.mean(all_successes):.1%}")
        print(f"Avg steps: {np.mean(all_steps):.1f}")
        print(f"Avg reward: {np.mean([sum(r) for r in all_rewards]):.4f}")
        print("="*70 + "\n")
        
        # Create visualization
        if render:
            visualize_trajectories(all_trajectories, all_actions, all_rewards, all_successes, all_crashes)
        
        env.close()
        

def visualize_trajectories(trajectories, actions, rewards, successes, crashes):
    """Visualize car trajectories"""
    num_eps = len(trajectories)
    cols = min(3, num_eps)
    rows = (num_eps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if num_eps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ep, (trajectory, action_seq, reward_seq, success, crashed) in enumerate(
            zip(trajectories, actions, rewards, successes, crashes)):
        ax = axes[ep]
        
        grid_size = 6
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(y=i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i - 0.5, color='lightgray', linewidth=0.5, alpha=0.5)
        
        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            
            # Draw path with color gradient
            for i in range(len(trajectory) - 1):
                color_val = i / max(len(trajectory) - 1, 1)
                color = plt.cm.coolwarm(color_val)
                ax.plot(trajectory[i:i+2, 1], trajectory[i:i+2, 0], 
                       color=color, linewidth=2.5, alpha=0.8, zorder=2)
            
            # Markers (without labels to save space)
            ax.plot(trajectory[0, 1], trajectory[0, 0], 'go', 
                   markersize=12, zorder=5)
            ax.plot(trajectory[-1, 1], trajectory[-1, 0], 'r*', 
                   markersize=18, zorder=5)
            
            steps = len(trajectory)
            # Action 0=GAS (move), Action 1=BRAKE (wait)
            action_str = ''.join(['G' if a == 0 else 'B' for a in action_seq[:10]])
            if len(action_seq) > 10:
                action_str += '...'
            
            episode_reward = np.sum(reward_seq)
            
            # Determine status: SUCCESS (reached goal), CRASH (collision), or TIMEOUT (max steps)
            if success:
                success_text = "✓ SUCCESS"
                title_color = 'green'
            elif crashed:
                success_text = "✗ CRASH"
                title_color = 'red'
            else:
                success_text = "✗ TIMEOUT"
                title_color = 'orange'
            
            title = f'Episode {ep+1}\n'
            title += f'{success_text} | Steps: {steps}, Reward: {episode_reward:.4f}\n'
            title += f'Actions: {action_str}'
            
            ax.set_title(title, fontsize=10, color=title_color)
        else:
            ax.text(3, 3, 'No movement', ha='center', va='center', fontsize=12)
            ax.set_title(f'Episode {ep+1} - Crashed')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.2)
    
    # Hide extra subplots
    for idx in range(num_eps, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Trained IC3Net Car Navigation\nBlue→Red: Start→End', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('car_trajectories_trained.png', dpi=100, bbox_inches='tight')
    print("✓ Saved car_trajectories_trained.png")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Track trained IC3Net policy')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--model', default='model.pt', type=str, help='Path to trained model')
    parser.add_argument('--no-render', action='store_true', help='Skip visualization')
    args = parser.parse_args()
    
    track_car_with_policy(args.episodes, args.model, render=not args.no_render)
