import sys
import time
import signal
import argparse
import warnings
import random

# Suppress gymnasium warnings to reduce terminal clutter
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

import numpy as np
import torch
import torch.multiprocessing as _mp
#import visdom
import matplotlib
# Try to use interactive backend with fallback
#try:
 #   matplotlib.use('TkAgg')  # Interactive backend for displaying plots
#except:
#    try:
#        matplotlib.use('Qt5Agg')  # Alternative interactive backend
#    except:
#        matplotlib.use('Agg')
matplotlib.use('Agg') # Fallback to non-interactive
import matplotlib.pyplot as plt
import data
from models import *
from comm_attention import AttentionCommNetMLP, HierarchicalAttentionCommNet
try:
    from comm import CommNetMLP
except ImportError:
    # Fallback keeps training runnable even if comm.py is absent in this workspace.
    CommNetMLP = AttentionCommNetMLP
from utils import *
from action_utils import parse_action_args, select_action, translate_action
from trainer import Trainer
from multi_processing import MultiProcessTrainer

# Guard access to possibly-removed backcompat attributes
if hasattr(torch, 'utils') and hasattr(torch.utils, 'backcompat'):
    if hasattr(torch.utils.backcompat, 'broadcast_warning'):
        try:
            torch.utils.backcompat.broadcast_warning.enabled = True
        except Exception:
            pass
    if hasattr(torch.utils.backcompat, 'keepdim_warning'):
        try:
            torch.utils.backcompat.keepdim_warning.enabled = True
        except Exception:
            pass

# Prefer setting default dtype (float64) instead of default tensor type string.
# Some newer PyTorch versions discourage `set_default_tensor_type(str)`.
try:
    torch.set_default_dtype(torch.float64)
except Exception:
    try:
        torch.set_default_tensor_type('torch.DoubleTensor')
    except Exception:
        pass


def _safe_end_display(env):
    try:
        env.end_display()
    except Exception:
        pass


def main():
    # Set spawn start method for multiprocessing when using CUDA or on Windows
    # This is required for CUDA + multiprocessing compatibility
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # already set
        pass

    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    # training
    # note: number of steps per epoch = epoch_size X batch_size x nprocesses
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=10,
                        help='number of update iterations in an epoch')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='number of steps before each update (per thread)')
    parser.add_argument('--nprocesses', type=int, default=16,
                        help='How many processes to run')
    # model
    parser.add_argument('--hid_size', default=64, type=int,
                        help='hidden layer size')
    parser.add_argument('--recurrent', action='store_true', default=False,
                        help='make the model recurrent in time')
    # optimization
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='gae (remove?)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed. Pass -1 for random seed') # TODO: works in thread?
    parser.add_argument('--normalize_rewards', action='store_true', default=False,
                        help='normalize rewards in each batch')
    parser.add_argument('--lrate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--entr', type=float, default=0,
                        help='entropy regularization coeff')
    parser.add_argument('--value_coeff', type=float, default=0.01,
                        help='coeff for value loss term')
    # environment
    parser.add_argument('--env_name', default="traffic_junction",
                        help='name of the environment (only traffic_junction supported)')
    parser.add_argument('--max_steps', default=20, type=int,
                        help='force to end the game after this many steps')
    parser.add_argument('--nactions', default='1', type=str,
                        help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
    parser.add_argument('--action_scale', default=1.0, type=float,
                        help='scale action output from model')
    # other
    parser.add_argument('--plot', action='store_true', default=False,
                        help='plot training progress')
    parser.add_argument('--plot_env', default='main', type=str,
                        help='plot env name')
    parser.add_argument('--log_path', default='', type=str,
                        help='path to save per-epoch training log (torch file)')
    parser.add_argument('--save', default='', type=str,
                        help='save the model after training')
    parser.add_argument('--save_every', default=100, type=int,
                        help='save the model after every n_th epoch')
    parser.add_argument('--load', default='', type=str,
                        help='load the model')
    parser.add_argument('--display', action="store_true", default=False,
                        help='Display environment state')
    parser.add_argument('--eval_every', default=50, type=int,
                        help='Run evaluation every N training epochs (0 disables evaluation)')
    parser.add_argument('--eval_epochs', default=20, type=int,
                        help='Number of evaluation epochs to average at each evaluation point')

    parser.add_argument('--random', action='store_true', default=False,
                        help="enable random model")

    # CommNet specific args
    parser.add_argument('--commnet', action='store_true', default=False,
                        help="enable commnet model")
    parser.add_argument('--ic3net', action='store_true', default=False,
                        help="enable commnet model")
    parser.add_argument('--nagents', type=int, default=1,
                        help="Number of agents (used in multiagent)")
    parser.add_argument('--comm_mode', type=str, default='avg',
                        help="Type of mode for communication tensor calculation [avg|sum]")
    parser.add_argument('--comm_passes', type=int, default=1,
                        help="Number of comm passes per step over the model")
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                        help="Whether communication should be there")
    parser.add_argument('--mean_ratio', default=1.0, type=float,
                        help='how much coooperative to do? 1.0 means fully cooperative')
    parser.add_argument('--rnn_type', default='MLP', type=str,
                        help='type of rnn to use. [LSTM|MLP]')
    parser.add_argument('--detach_gap', default=10000, type=int,
                        help='detach hidden state and cell state for rnns at this interval.'
                        + ' Default 10000 (very high)')
    parser.add_argument('--comm_init', default='uniform', type=str,
                        help='how to initialise comm weights [uniform|zeros]')
    parser.add_argument('--hard_attn', default=False, action='store_true',
                        help='Whether to use hard attention: action - talk|silent')
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                        help='Whether to always talk, sanity check for hard attention.')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                        help='Whether to multipy log porb for each chosen action with advantages')
    parser.add_argument('--share_weights', default=False, action='store_true',
                        help='Share weights for hops')
    parser.add_argument('--use_agent_attn', default=False, action='store_true',
                        help='Apply multi-head self-attention over agent hidden states')
    parser.add_argument('--attn_heads', type=int, default=4,
                        help='Number of heads for agent self-attention')
    parser.add_argument('--flash', default=False, action='store_true',
                        help='Use FlashAttention-style agent attention over hidden states')
    parser.add_argument('--flash_attn_dropout', type=float, default=0.0,
                        help='Dropout used inside flash attention (during training only)')
    parser.add_argument('--flash_block_size', type=int, default=32,
                        help='Block size for Flash Attention tiling (smaller = less memory, larger = faster)')
    parser.add_argument('--flash_gate_init', type=float, default=0.1,
                        help='Initial residual gate value for flash attention (before sigmoid)')
    parser.add_argument('--mamba', default=False, action='store_true',
                        help='Use a Mamba-style recurrent block for temporal memory')
    parser.add_argument('--mamba_dropout', type=float, default=0.1,
                        help='Dropout rate for Mamba block')
    parser.add_argument('--hier_local_attn', default=False, action='store_true',
                        help='Use hierarchical local attention communication model')
    parser.add_argument('--team_size', type=int, default=5,
                        help='Team size for hierarchical local attention')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping (0 to disable)', required=False)

    # Traffic Junction environment arguments
    env_group = parser.add_argument_group('Traffic Junction task')
    env_group.add_argument('--dim', type=int, default=6,
                          help="Dimension of box (i.e length of road)")
    env_group.add_argument('--vision', type=int, default=1,
                          help="Vision of car")
    env_group.add_argument('--add_rate_min', type=float, default=0.05,
                          help="rate at which to add car (till curr. start)")
    env_group.add_argument('--add_rate_max', type=float, default=0.2,
                          help=" max rate at which to add car")
    env_group.add_argument('--curr_start', type=float, default=0,
                          help="start making harder after this many epochs [0]")
    env_group.add_argument('--curr_end', type=float, default=0,
                          help="when to make the game hardest [0]")
    env_group.add_argument('--crash_penalty', type=float, default=-2.0,
                          help="penalty applied to each agent on collision")
    env_group.add_argument('--terminal_reward', type=float, default=15.0,
                          help="reward applied when an agent reaches destination")
    env_group.add_argument('--difficulty', type=str, default='easy',
                          help="Difficulty level, easy|medium|hard")
    env_group.add_argument('--vocab_type', type=str, default='bool',
                          help="Type of location vector to use, bool|scalar")

    args = parser.parse_args()

    if args.ic3net:
        args.commnet = 1 # Agent communication enabled
        args.hard_attn = 1 # Agent can choose to communicate or not [move_action, comm_action]
        args.mean_ratio = 1 # fully cooperative
        args.comm_action_one = False # not always communicate

    enabled_alt_models = int(args.flash) + int(args.mamba) + int(args.hier_local_attn)
    if enabled_alt_models > 1:
        raise ValueError('Choose only one of --flash, --mamba, or --hier_local_attn')

    if args.flash:
        args.commnet = 1
        args.use_agent_attn = False

    if args.mamba:
        args.commnet = 1
        args.recurrent = True
        args.rnn_type = 'MAMBA'

    if args.hier_local_attn:
        args.commnet = 1
        args.recurrent = False

    # Set friendly agents count (traffic junction only has friendly cars)
    args.nfriendly = args.nagents

    env = data.init(args.env_name, args, False)

    num_inputs = env.observation_dim
    args.num_actions = env.num_actions

    # Multi-action
    if not isinstance(args.num_actions, (list, tuple)): # single action case
        args.num_actions = [args.num_actions]
    args.dim_actions = env.dim_actions
    args.num_inputs = num_inputs

    # Hard attention
    if args.hard_attn and args.commnet:
        # add comm_action as last dim in actions
        args.num_actions = [*args.num_actions, 2]
        args.dim_actions = env.dim_actions + 1

    # Recurrence
    if args.commnet and (args.recurrent or args.rnn_type in ['LSTM', 'MAMBA']):
        args.recurrent = True
        if args.rnn_type not in ['LSTM', 'MAMBA']:
            args.rnn_type = 'LSTM'


    parse_action_args(args)

    if args.seed == -1:
        args.seed = np.random.randint(0,10000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device (GPU if available, else CPU)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {args.device}')
    if args.device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(args)


    if args.commnet:
        if args.hier_local_attn:
            policy_net = HierarchicalAttentionCommNet(args, num_inputs, team_size=args.team_size)
        elif args.flash:
            policy_net = AttentionCommNetMLP(args, num_inputs)
        else:
            policy_net = CommNetMLP(args, num_inputs)
    elif args.random:
        policy_net = Random(args, num_inputs)
    elif args.recurrent:
        policy_net = RNN(args, num_inputs)
    else:
        policy_net = MLP(args, num_inputs)

    # Move model to device
    policy_net = policy_net.to(args.device)

    if not args.display:
        display_models([policy_net])

    # share parameters among threads, but not gradients
    for p in policy_net.parameters():
        p.data.share_memory_()

    if args.nprocesses > 1:
        trainer = MultiProcessTrainer(args, lambda: Trainer(args, policy_net, data.init(args.env_name, args)))
    else:
        trainer = Trainer(args, policy_net, data.init(args.env_name, args))

    eval_trainer = Trainer(args, policy_net, data.init(args.env_name, args))

    disp_trainer = Trainer(args, policy_net, data.init(args.env_name, args, False))
    disp_trainer.display = False  # Don't use curses rendering (doesn't work on Windows)
    
    def visualize_episode(policy_net, env_wrapper, epoch, args):
        """Run a fresh episode with the trained policy and visualize it step-by-step"""
        print(f'\n{"="*50}')
        print(f'Visualization Episode - Epoch {epoch+1}')
        print(f'{"="*50}')
        
        # Unwrap to get the actual environment
        env = env_wrapper.env
        while hasattr(env, 'env'):
            env = env.env
        
        # Get grid dimensions
        grid_size = env.dim if hasattr(env, 'dim') else 6
        
        # Reset environment
        state = env_wrapper.reset(epoch)
        
        # Initialize hidden state for LSTM
        if args.recurrent:
            prev_hid = policy_net.init_hidden(batch_size=1)
        
        # Initialize info dict - comm_action is required for IC3Net
        info = {}
        if args.commnet:
            info['comm_action'] = np.zeros(args.nagents, dtype=int)
        
        # Collect trajectory
        positions = []
        actions_taken = []
        rewards_list = []
        destinations = []  # Track destination as it may change when car is added
        
        total_reward = 0
        done = False
        step_count = 0
        
        # Record initial position
        if hasattr(env, 'car_loc') and len(env.car_loc) > 0:
            positions.append([tuple(pos) for pos in env.car_loc])
        
        car_spawned_at_step = None
        completed_at_step = None
        completed_cars = []  # Track which cars completed and when
        crashed = False
        
        for t in range(args.max_steps):
            # Track current destination (updates when car is added)
            current_dest = None
            if hasattr(env, 'chosen_path') and len(env.chosen_path) > 0:
                if isinstance(env.chosen_path[0], np.ndarray) and len(env.chosen_path[0]) > 0:
                    current_dest = tuple(env.chosen_path[0][-1])
            destinations.append(current_dest)
            # Get action from policy
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(args.device)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
                state_tensor = state_tensor.to(args.device)
            else:
                state_tensor = torch.tensor(state).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                if args.recurrent:
                    x = [state_tensor, prev_hid]
                    action_out, value, prev_hid = policy_net(x, info)
                else:
                    x = state_tensor
                    action_out, value = policy_net(x, info)
            
            # Select and translate action
            action = select_action(args, action_out)
            action, actual = translate_action(args, env_wrapper, action)
            
            # Record action
            if isinstance(actual, (list, np.ndarray)) and len(actual) > 0:
                actions_taken.append(actual[0])
            
            # Update comm action for next step
            if args.hard_attn and args.commnet:
                comm_action = np.array(action[-1]) if not args.comm_action_one else np.ones(args.nagents, dtype=int)
                info['comm_action'] = comm_action
            
            # Step environment (wrapper returns 4 values: obs, reward, done, info)
            # The wrapper internally handles Gymnasium's 5-value return
            next_state, reward, done, env_info = env_wrapper.step(actual)
            
            # Check for completion/crash in this step
            if 'is_completed' in env_info and np.any(env_info['is_completed'] == 1):
                # Track which cars completed in this step
                for car_idx in range(len(env_info['is_completed'])):
                    if env_info['is_completed'][car_idx] == 1 and car_idx not in [c[0] for c in completed_cars]:
                        completed_cars.append((car_idx, t))
                        print(f"Step {t}: *** CAR {car_idx} COMPLETED ROUTE! ***")
                
                # Set completed_at_step to first completion
                if completed_at_step is None:
                    completed_at_step = t
                    print(f"           done={done}, episode_over={env.episode_over if hasattr(env, 'episode_over') else '?'}")
                    print(f"           cars_in_sys={env.cars_in_sys if hasattr(env, 'cars_in_sys') else '?'}")
            
            if hasattr(env, 'has_failed') and env.has_failed == 1:
                if not crashed:
                    crashed = True
                    print(f"Step {t}: *** CAR CRASHED! ***")
            
            # Track car spawning and movement
            cars_now = env.cars_in_sys if hasattr(env, 'cars_in_sys') else 0
            if cars_now > 0 and car_spawned_at_step is None:
                car_spawned_at_step = t
                car_pos = env.car_loc[0] if hasattr(env, 'car_loc') else None
                print(f"Step {t}: *** CAR SPAWNED at position {tuple(car_pos)} ***")
            
            # Preserve comm_action in info for next iteration
            if args.commnet:
                if 'comm_action' not in env_info:
                    env_info['comm_action'] = info.get('comm_action', np.zeros(args.nagents, dtype=int))
            info = env_info
            
            total_reward += reward.sum()
            rewards_list.append(reward.sum())
            state = next_state
            step_count = t + 1
            
            # Record new position only if car is still active (not completed/crashed)
            if hasattr(env, 'car_loc') and len(env.car_loc) > 0:
                # Check if car is alive (not at reset position 0,0)
                car_alive = hasattr(env, 'alive_mask') and np.any(env.alive_mask == 1)
                if car_alive and cars_now > 0:
                    positions.append([tuple(pos) for pos in env.car_loc])
            
            # Let environment handle termination naturally - don't force break
            if done:
                break
        
        # Get final destination (use the last valid one)
        dest_pos = None
        for d in reversed(destinations):
            if d is not None:
                dest_pos = d
                break
        
        # Get actual grid size from environment (easy difficulty adds 1 to dim)
        actual_grid_size = env.dims[0] if hasattr(env, 'dims') else grid_size
        
        # Now visualize the collected trajectory
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        
        for step_idx, pos_list in enumerate(positions):
            ax.clear()
            ax.set_xlim(-0.5, actual_grid_size - 0.5)
            ax.set_ylim(-1.2, actual_grid_size - 0.5)  # Minimal space
            ax.set_aspect('equal')
            # Show actual step number (step_idx) vs max allowed steps
            ax.set_title(f'Traffic Junction - Epoch {epoch+1} - Step {step_idx}/{args.max_steps}', 
                        fontsize=10, weight='bold', pad=5)
            ax.set_xticks(range(actual_grid_size))
            ax.set_yticks(range(actual_grid_size))
            ax.invert_yaxis()  # Invert y-axis so (0,0) is at top-left like array indexing
            
            # Draw background (grass/outside area)
            for i in range(actual_grid_size):
                for j in range(actual_grid_size):
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                        facecolor='lightgreen', 
                                        edgecolor='darkgreen', linewidth=0.5, alpha=0.3)
                    ax.add_patch(rect)
            
            # Draw complete crossroads (vertical and horizontal roads)
            mid = actual_grid_size // 2
            # Vertical road (TOP to BOTTOM)
            for i in range(actual_grid_size):
                rect = plt.Rectangle((mid-0.5, i-0.5), 1, 1, 
                                    facecolor='gray', edgecolor='white', linewidth=1)
                ax.add_patch(rect)
            # Horizontal road (LEFT to RIGHT)
            for j in range(actual_grid_size):
                rect = plt.Rectangle((j-0.5, mid-0.5), 1, 1, 
                                    facecolor='gray', edgecolor='white', linewidth=1)
                ax.add_patch(rect)
            
            # Draw junction center (intersection) with special highlight
            junction = plt.Rectangle((mid-0.5, mid-0.5), 1, 1, 
                                    facecolor='yellow', edgecolor='orange', 
                                    linewidth=2, alpha=0.5)
            ax.add_patch(junction)
            
            # Draw road markings (dashed center lines)
            # Vertical road center line
            for i in range(actual_grid_size):
                if i != mid:  # Skip junction
                    ax.plot([mid, mid], [i-0.3, i+0.3], 'w--', linewidth=1.5, alpha=0.7)
            # Horizontal road center line  
            for j in range(actual_grid_size):
                if j != mid:  # Skip junction
                    ax.plot([j-0.3, j+0.3], [mid, mid], 'w--', linewidth=1.5, alpha=0.7)
            
            # Draw trajectory path (show the route the car is following)
            if step_idx > 0:
                for i in range(step_idx):
                    prev_pos = positions[i][0]
                    next_pos = positions[i+1][0] if i+1 < len(positions) else prev_pos
                    # positions are (row, col), plot as (col, row) since y-axis is inverted
                    ax.plot([prev_pos[1], next_pos[1]], 
                           [prev_pos[0], next_pos[0]], 
                           'cyan', alpha=0.5, linewidth=3, linestyle='-', zorder=3)
                    # Add directional arrow
                    if i == step_idx - 1 and step_idx > 0:
                        dx = next_pos[1] - prev_pos[1]
                        dy = next_pos[0] - prev_pos[0]
                        if dx != 0 or dy != 0:
                            ax.arrow(prev_pos[1], prev_pos[0], dx*0.4, dy*0.4,
                                   head_width=0.2, head_length=0.15, fc='cyan', ec='cyan', 
                                   alpha=0.7, zorder=3)
            
            # Draw car at current position
            for car_idx, car_pos in enumerate(pos_list):
                # Turn green after car completes (at or after completed_at_step)
                is_completed_now = completed_at_step is not None and step_idx >= completed_at_step
                is_crashed = crashed and step_idx >= 0  # Show red if crashed
                
                if is_crashed:
                    color = 'red'
                    edge_color = 'darkred'
                elif is_completed_now:
                    color = 'green'
                    edge_color = 'darkgreen'
                else:
                    color = 'blue'
                    edge_color = 'darkblue'
                
                # Draw car as larger circle with border
                circle = plt.Circle((car_pos[1], car_pos[0]), 
                                  0.35, facecolor=color, edgecolor=edge_color, 
                                  linewidth=2.5, zorder=10)
                ax.add_patch(circle)
                
                # Add car label
                ax.text(car_pos[1], car_pos[0], 'C', 
                       ha='center', va='center', fontsize=7, weight='bold', 
                       color='white', zorder=11)
                
                # Action label is shown in info box, not on grid
            
            # Show destination (start point)
            if dest_pos:
                # dest_pos is (row, col), plot as (col, row) since y-axis is inverted
                dest_x = dest_pos[1]
                dest_y = dest_pos[0]
                # Draw larger star
                ax.plot(dest_x, dest_y, 'r*', markersize=40, zorder=5, 
                       markeredgewidth=2.5, markeredgecolor='darkred')
                # Add label below destination
                ax.text(dest_x, dest_y + 0.4, 'FINISH', ha='center', 
                       fontsize=7, weight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                               edgecolor='red', linewidth=1, alpha=0.9))
                
                # Show start position (first position in route)
                if positions and len(positions) > 0:
                    start_pos = positions[0][0]
                    if start_pos[0] != dest_pos[0] or start_pos[1] != dest_pos[1]:
                        ax.plot(start_pos[1], start_pos[0], 'g^', markersize=25, zorder=5,
                               markeredgewidth=2, markeredgecolor='darkgreen')
                        ax.text(start_pos[1], start_pos[0] - 0.4, 'START', ha='center',
                               fontsize=6, weight='bold', color='green',
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='lightgreen',
                                       edgecolor='darkgreen', linewidth=0.8, alpha=0.9))
            
            # Show completion/crash status banner (compact)
            if crashed and step_idx > 0:
                ax.text(actual_grid_size/2, -1.0, 'CRASHED', 
                       ha='center', fontsize=8, weight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', 
                               edgecolor='red', linewidth=1.5, alpha=0.95))
            elif completed_at_step is not None and step_idx >= completed_at_step:
                ax.text(actual_grid_size/2, -1.0, 'DONE', 
                       ha='center', fontsize=8, weight='bold', color='green',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', 
                               edgecolor='darkgreen', linewidth=1.5, alpha=0.95))
            
            # Add compact info box
            info_parts = [f"Step {step_idx}/{args.max_steps}"]
            if step_idx < len(actions_taken):
                action_text = 'GAS' if actions_taken[step_idx] == 0 else 'BRAKE'
                info_parts.append(action_text)
            if car_spawned_at_step is not None:
                info_parts.append(f"T+{step_idx - car_spawned_at_step}")
            info_text = " | ".join(info_parts)
            
            ax.text(actual_grid_size/2, -0.6, info_text, ha='center',
                   fontsize=7, verticalalignment='top', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', 
                           edgecolor='darkblue', linewidth=1, alpha=0.9))
            
            plt.draw()
            
            # Adjust pause time: slow before completion, fast after
            if completed_at_step is not None and step_idx > completed_at_step:
                # After completion: very fast (skip through empty frames)
                plt.pause(0.05)
            elif completed_at_step is not None and step_idx == completed_at_step:
                # At completion: pause longer to emphasize success
                plt.pause(2.0)
            else:
                # Normal movement: slower for visibility
                plt.pause(0.8)
        
        plt.ioff()
        # Save plot to file
        plot_filename = f'episode_visualization_epoch_{epoch+1}.png'
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Determine success and print final summary
        natural_end = done and not crashed and car_spawned_at_step is not None
        success = (completed_at_step is not None or natural_end) and not crashed
        
        print(f"\n{'='*50}")
        print(f'Episode Length: {step_count} steps')
        print(f'Total Reward: {total_reward:.2f}')
        success_text = "Yes ✓" if success else "No ✗"
        print(f'Success: {success_text}')
        print(f'Visualization saved: {plot_filename}')
        if success:
            print(f'  Car completed route without crashes!')
        elif crashed:
            print(f'  Car crashed!')
        else:
            print(f'  Car did not complete route')
        print(f'{"="*50}\n')
    
    def disp(epoch):
        # Run a fresh visualization episode with the current policy
        visualize_episode(policy_net, disp_trainer.env, epoch, args)

    log = dict()
    log['epoch'] = LogField(list(), False, None, None)
    log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')  # Averaged over epoch
    log['success'] = LogField(list(), True, 'epoch', None)  # Per-episode metric, no division
    log['completion_rate'] = LogField(list(), True, 'epoch', 'num_episodes')  # Averaged over epoch
    log['total_crashes'] = LogField(list(), True, 'epoch', 'num_episodes')  # Averaged over epoch
    log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')  # Averaged over epoch
    log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
    log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
    log['eval_epoch'] = LogField(list(), False, None, None)
    log['eval_reward'] = LogField(list(), False, 'eval_epoch', 'num_episodes')
    log['eval_success'] = LogField(list(), False, 'eval_epoch', None)
    log['eval_completion_rate'] = LogField(list(), False, 'eval_epoch', 'num_episodes')
    log['eval_total_crashes'] = LogField(list(), False, 'eval_epoch', 'num_episodes')
    log['eval_steps_taken'] = LogField(list(), False, 'eval_epoch', 'num_episodes')

    if args.plot:
        #vis = visdom.Visdom(env=args.plot_env)
        args.plot = False
    
    def run(num_epochs):
        print('Starting training now...')
        for ep in range(num_epochs):
            epoch_begin_time = time.time()
            stat = dict()
            # Accumulate stats across all epoch_size episodes for averaging
            for n in range(args.epoch_size):
                if n == args.epoch_size - 1 and args.display:
                    trainer.display = True
                s = trainer.train_batch(ep)
                # Accumulate statistics from each episode in the epoch
                merge_stat(s, stat)
                trainer.display = False
            
            # Display visualization after each epoch using disp_trainer
            # Visualization disabled to avoid interference during multiple training runs
            
            epoch_time = time.time() - epoch_begin_time
            epoch = len(log['epoch'].data) + 1 #number of epochs completed
            
            for k, v in log.items():
                if k == 'epoch':
                    v.data.append(epoch)
                elif k.startswith('eval_'):
                    continue
                else:
                    if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                        stat[k] = stat[k] / stat[v.divide_by]
                    v.data.append(stat.get(k, 0))

            eval_stat = None
            if args.eval_every > 0 and args.eval_epochs > 0 and epoch % args.eval_every == 0:
                py_rng_state = random.getstate()
                np_rng_state = np.random.get_state()
                torch_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                try:
                    eval_stat = dict()
                    for _ in range(args.eval_epochs):
                        s = eval_trainer.evaluate_batch(ep)
                        merge_stat(s, eval_stat)

                    for metric_key in ['reward', 'completion_rate', 'total_crashes', 'steps_taken']:
                        if metric_key in eval_stat and eval_stat.get('num_episodes', 0) > 0:
                            eval_stat[metric_key] = eval_stat[metric_key] / eval_stat['num_episodes']

                    log['eval_epoch'].data.append(epoch)
                    log['eval_reward'].data.append(eval_stat.get('reward', 0))
                    log['eval_success'].data.append(eval_stat.get('success', 0))
                    log['eval_completion_rate'].data.append(eval_stat.get('completion_rate', 0))
                    log['eval_total_crashes'].data.append(eval_stat.get('total_crashes', 0))
                    log['eval_steps_taken'].data.append(eval_stat.get('steps_taken', 0))
                finally:
                    random.setstate(py_rng_state)
                    np.random.set_state(np_rng_state)
                    torch.set_rng_state(torch_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state_all(cuda_rng_state)

            # Optionally save per-epoch log to disk for later analysis/plotting
            if args.log_path:
                try:
                    import torch as _torch
                    _torch.save({'log': log}, args.log_path)
                except Exception:
                    pass

            np.set_printoptions(precision=2)

            # Convert numpy arrays to scalars for printing
            if isinstance(stat['reward'], np.ndarray):
                reward_array = stat['reward']
                reward_val = float(reward_array.mean()) if reward_array.size > 1 else float(reward_array)
                # Extract per-agent rewards (first 5 agents)
                per_agent_rewards = reward_array[:min(5, len(reward_array))]
            else:
                reward_val = stat['reward']
                per_agent_rewards = np.array([reward_val])
            
            # Extract and format key statistics
            success_val = 0.0
            if 'success' in stat.keys():
                success_val = float(stat['success']) if isinstance(stat['success'], np.ndarray) else stat['success']
            
            steps_val = 0.0
            if 'steps_taken' in stat.keys():
                steps_val = float(stat['steps_taken'])
            
            add_rate_val = 0.0
            if 'add_rate' in stat.keys():
                if isinstance(stat['add_rate'], np.ndarray):
                    add_rate_val = float(stat['add_rate'].mean()) if stat['add_rate'].size > 1 else float(stat['add_rate'])
                else:
                    add_rate_val = stat['add_rate']
            
            # Print clear, formatted output (reduce frequency to avoid terminal slowdown)
            # Print every epoch for first 10, then every 10 epochs, then every 50 epochs after 100
            should_print = (epoch <= 10 or epoch % 10 == 0 or ep == num_epochs - 1)
            if epoch > 100:
                should_print = (epoch % 50 == 0 or ep == num_epochs - 1)
            
            if should_print:
                print('=' * 70)
                print('Epoch {:4d} | Time: {:6.2f}s'.format(epoch, epoch_time))
                print('-' * 70)
                
                # Display per-agent rewards (epoch-averaged across epoch_size episodes)
                if isinstance(stat.get('reward', None), np.ndarray) and len(stat['reward']) > 0:
                    per_agent_str = ' '.join(['{:6.2f}'.format(r) for r in stat['reward']])
                    print('  Per-Agent Rewards (epoch avg): [{}]'.format(per_agent_str))
                
                print('  Avg Reward (epoch):  {:7.2f}  '.format(reward_val))
                
                # Show completion rate (fraction of agents that reached their goal, epoch-averaged)
                if 'completion_rate' in stat.keys():
                    completion_val = float(stat['completion_rate']) if isinstance(stat['completion_rate'], np.ndarray) else stat['completion_rate']
                    print('  Avg Completion:      {:6.1f}%  (epoch avg)'.format(completion_val * 100))
                
                # Show average crashes per episode in the epoch
                if 'total_crashes' in stat.keys():
                    crashes_val = float(stat['total_crashes']) if isinstance(stat['total_crashes'], np.ndarray) else float(stat['total_crashes'])
                    print('  Avg Crashes:         {:7.2f}  (epoch avg per episode)'.format(crashes_val))
                
                # Show average episode length in the epoch
                if 'steps_taken' in stat.keys() and steps_val > 0:
                    print('  Avg Episode Steps:   {:7.2f}  (epoch avg)'.format(steps_val))
                
                # MPE-specific metrics: coverage distance
                if 'min_dist' in stat.keys():
                    min_dist_val = float(stat['min_dist']) if isinstance(stat['min_dist'], np.ndarray) else stat['min_dist']
                    print('  Min Landmark Dist: {:5.3f}  '.format(min_dist_val))
                
                if 'avg_dist' in stat.keys():
                    avg_dist_val = float(stat['avg_dist']) if isinstance(stat['avg_dist'], np.ndarray) else stat['avg_dist']
                    print('  Avg Landmark Dist: {:5.3f}  '.format(avg_dist_val))
                
                # Collision tracking for MPE
                if 'collisions' in stat.keys():
                    coll_val = float(stat['collisions']) if isinstance(stat['collisions'], np.ndarray) else stat['collisions']
                    print('  Collisions:     {:7.2f}  '.format(coll_val))
                
                # Communication percentage (all environments with IC3Net, epoch-averaged)
                if 'comm_action' in stat.keys():
                    comm_vals = stat['comm_action']
                    if isinstance(comm_vals, np.ndarray) and len(comm_vals) > 0:
                        print('  Communication:      {:6.1f}%  (epoch avg)'.format(float(comm_vals.mean()) * 100))

                if eval_stat is not None:
                    eval_reward_val = eval_stat.get('reward', 0)
                    if isinstance(eval_reward_val, np.ndarray):
                        eval_reward_val = float(np.mean(eval_reward_val)) if eval_reward_val.size > 0 else 0.0
                    eval_success_val = eval_stat.get('success', 0)
                    eval_completion_val = eval_stat.get('completion_rate', 0)
                    eval_crashes_val = eval_stat.get('total_crashes', 0)
                    print('  Eval over {:2d} epochs: Reward {:7.2f} | Success {:6.1f}% | Completion {:6.1f}% | Crashes {:7.2f}'.format(
                        args.eval_epochs,
                        float(eval_reward_val),
                        float(eval_success_val) * 100,
                        float(eval_completion_val) * 100,
                        float(eval_crashes_val)
                    ))
                print('=' * 70)

            if args.plot:
                for k, v in log.items():
                    if v.plot and len(v.data) > 0:
                        vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                        win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

            if args.save_every and ep and args.save != '' and ep % args.save_every == 0:
                # fname, ext = args.save.split('.')
                # save(fname + '_' + str(ep) + '.' + ext)
                save(args.save + '_' + str(ep))

            if args.save != '':
                save(args.save)

    def save(path):
        d = dict()
        d['policy_net'] = policy_net.state_dict()
        d['log'] = log
        d['trainer'] = trainer.state_dict()
        torch.save(d, path)

    def load(path):
        d = torch.load(path, map_location=args.device)
        # log.clear()
        policy_net.load_state_dict(d['policy_net'])
        log.update(d['log'])
        trainer.load_state_dict(d['trainer'])

    def signal_handler(signum, frame):
        print('You pressed Ctrl+C! Exiting gracefully.')
        if args.display:
            _safe_end_display(env)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.load != '':
        load(args.load)
   
    run(args.num_epochs)
    if args.display:
        _safe_end_display(env)

    if args.save != '':
        save(args.save)

    if sys.flags.interactive == 0 and args.nprocesses > 1:
        trainer.quit()
        import os
        os._exit(0)


if __name__ == '__main__':
    main()
