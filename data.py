import sys
import torch
import gymnasium as gym
from gymnasium.wrappers import PassiveEnvChecker
import ic3net_envs
from env_wrappers import *
try:
    from pettingzoo_multiwalker import MultiWalkerDiscretizedWrapper
except ImportError:
    MultiWalkerDiscretizedWrapper = None
from pettingzoo_mpe import MPESimpleSpreadWrapper


def _ensure_traffic_junction_registered():
    """Ensure TrafficJunction custom env is registered with Gymnasium."""
    try:
        gym.spec('TrafficJunction-v0')
        return
    except Exception:
        pass

    try:
        import ic3net_envs.ic3net_envs  # noqa: F401
    except Exception:
        try:
            import ic3net_envs  # noqa: F401
        except Exception:
            pass

    try:
        gym.spec('TrafficJunction-v0')
    except Exception as e:
        raise RuntimeError(
            "TrafficJunction-v0 is not registered. Install with `pip install -e ./ic3net_envs` "
            "in the same Python environment used to run main.py"
        ) from e

def _unwrap_env(env):
    """Unwrap gymnasium wrappers to access the underlying custom env."""
    while hasattr(env, 'env'):
        env = env.env
    return env

def init(env_name, args, final_init=True):
    if env_name == 'traffic_junction':
        _ensure_traffic_junction_registered()
        env = gym.make('TrafficJunction-v0')
        
        # Remove PassiveEnvChecker wrapper if it exists (custom obs format doesn't match standard)
        while isinstance(env, PassiveEnvChecker):
            env = env.env
        
        unwrapped = _unwrap_env(env)
        if args.display:
            if hasattr(unwrapped, 'init_curses'):
                unwrapped.init_curses()
        
        # Set number of cars from args
        try:
            agent_count = getattr(args, 'nfriendly', None) or getattr(args, 'nagents', 1)
            unwrapped.ncar = int(agent_count)
        except Exception:
            pass

        # Initialize environment
        unwrapped.multi_agent_init(args)
        device = getattr(args, 'device', torch.device('cpu'))
        env = GymWrapper(env, device=device)
    
    elif env_name == 'multiwalker':
        # PettingZoo MultiWalker: Cooperative bipedal walkers carrying a package
        # Much more challenging than traffic junction!
        if MultiWalkerDiscretizedWrapper is None:
            raise RuntimeError("MultiWalker requires box2d-py (install failed). Use 'mpe_spread' instead.")
        
        agent_count = getattr(args, 'nfriendly', None) or getattr(args, 'nagents', 3)
        max_steps = getattr(args, 'max_steps', 500)
        device = getattr(args, 'device', torch.device('cpu'))
        
        print(f"Initializing MultiWalker with {agent_count} walkers, max_steps={max_steps}")
        
        env = MultiWalkerDiscretizedWrapper(
            n_walkers=agent_count,
            max_cycles=max_steps,
            shared_reward=True,  # Cooperative task
            terminate_on_fall=False,  # Continue episode even if walker falls
            remove_on_fall=True,  # Remove fallen walker
            forward_reward=1.0,
            fall_reward=-10.0,
            terminate_reward=-100.0,
            device=device
        )
    
    elif env_name == 'mpe_spread':
        # PettingZoo MPE Simple Spread: Cooperative landmark coverage
        # N agents must spread to cover N landmarks, avoiding collisions
        # Great for testing communication effectiveness!
        agent_count = getattr(args, 'nfriendly', None) or getattr(args, 'nagents', 3)
        max_steps = getattr(args, 'max_steps', 25)
        device = getattr(args, 'device', torch.device('cpu'))
        
        print(f"Initializing MPE Simple Spread with {agent_count} agents, max_steps={max_steps}")
        
        env = MPESimpleSpreadWrapper(
            n_agents=agent_count,
            max_cycles=max_steps,
            continuous_actions=False,  # Use discrete actions for IC3Net
            device=device
        )
    
    else:
        raise RuntimeError(f"Unsupported environment: {env_name}. Supported: 'traffic_junction', 'multiwalker', 'mpe_spread'")

    return env

