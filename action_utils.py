import numpy as np
import torch
from torch.autograd import Variable


def _safe_multinomial_sample(probs):
    """Sample from probs safely by removing invalid values and normalizing rows."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)

    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    row_sums = probs.sum(dim=-1, keepdim=True)
    valid_rows = row_sums.squeeze(-1) > 0
    if not torch.all(valid_rows):
        # If a row collapsed to all zeros, sample uniformly for that row.
        probs = probs.clone()
        probs[~valid_rows] = 1.0
        row_sums = probs.sum(dim=-1, keepdim=True)

    probs = probs / row_sums
    sample = torch.multinomial(probs, 1).detach()

    if squeeze_back:
        sample = sample.squeeze(0)
    return sample

def parse_action_args(args):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        args.continuous = False
        # assert args.dim_actions == 1
        # support multi action
        args.naction_heads = [int(args.num_actions[i]) for i in range(args.dim_actions)]
    else:
        # environment takes continuous action
        actions_heads = args.nactions.split(':')
        if len(actions_heads) == 1 and int(actions_heads[0]) == 1:
            args.continuous = True
        elif len(actions_heads) == 1 and int(actions_heads[0]) > 1:
            args.continuous = False
            args.naction_heads = [int(actions_heads[0]) for _ in range(args.dim_actions)]
        elif len(actions_heads) > 1:
            args.continuous = False
            args.naction_heads = [int(i) for i in actions_heads]
        else:
            raise RuntimeError("--nactions wrong format!")


def select_action(args, action_out, eval_mode=False):
    # eval_mode uses same stochastic sampling as training (no argmax)
    # This preserves IC3Net's learned communication distribution
    # Gradients are disabled at the call site (torch.no_grad) instead
    if args.continuous:
        action_mean, _, action_std = action_out
        action = torch.normal(action_mean, action_std)
        return action.detach()
    else:
        log_p_a = action_out
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([_safe_multinomial_sample(x) for x in p]) for p in p_a])
        return ret

def translate_action(args, env, action):
    if args.num_actions[0] > 0:
        # environment takes discrete action
        # action from select_action has shape (dim_actions, batch_size, nagents, 1)
        # We need to extract one action per agent
        nagents = args.nagents
        
        # Determine how many environment actions (vs communication actions)
        # If hard_attn is enabled, the last dimension is for communication
        num_env_actions = len(action)
        if args.hard_attn and args.commnet:
            num_env_actions -= 1  # Exclude the communication action dimension
        
        # Build action array for trainer: list of (dim_actions, nagents)
        # This preserves structure for trainer's compute_grad
        action_array = []
        
        # Extract all actions (env + comm) in order
        for dim_idx in range(len(action)):
            action_tensor = action[dim_idx]  # shape (batch_size, nagents, 1)
            # For batch_size=1 (single episode), take first batch
            action_tensor = action_tensor[0]  # shape (nagents, 1)
            agent_actions = []
            # For each agent, extract the action value
            for agent_idx in range(nagents):
                val = action_tensor[agent_idx].squeeze().data.cpu().numpy()
                if isinstance(val, np.ndarray):
                    val = float(val.item()) if val.size == 1 else float(val)
                agent_actions.append(int(val))
            action_array.append(agent_actions)
        
        # Extract only environment actions for env.step()
        env_action_array = action_array[:num_env_actions]
        actual = np.array([a for actions_per_dim in env_action_array for a in actions_per_dim], dtype=int)
        
        # Return actions in structure (dim_actions, nagents) for trainer, actual array for env
        return action_array, actual
    else:
        if args.continuous:
            action = action.data[0].cpu().numpy()
            cp_action = action.copy()
            # clip and scale action to correct range
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                cp_action[i] = cp_action[i] * args.action_scale
                cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
                cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
            return action, cp_action
        else:
            actual = np.zeros(len(action))
            for i in range(len(action)):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                actual[i] = action[i].data.squeeze().cpu()[0] * (high - low) / (args.naction_heads[i] - 1) + low
            action = [x.squeeze().data.cpu()[0] for x in action]
            return action, actual
