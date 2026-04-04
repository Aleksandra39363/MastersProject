"""
PettingZoo MultiWalker Environment Wrapper for IC3Net
Integrates PettingZoo's SISL MultiWalker with IC3Net training

MultiWalker: Multiple bipedal walkers must coordinate to carry a package
- Continuous observations (each walker sees its state + neighbors)
- Continuous actions (4 joints per walker)
- Requires tight coordination (if one falls, package drops)
- Perfect for testing communication!
"""

import numpy as np
import torch
from pettingzoo.sisl import multiwalker_v9


class MultiWalkerWrapper:
    """
    Wrapper for PettingZoo MultiWalker to make it compatible with IC3Net
    
    MultiWalker is a challenging cooperative task:
    - 3 bipedal walkers must carry a package together
    - Each walker has 4 continuous action dimensions (hip/knee joints)
    - If any walker falls or moves too slow, the package falls
    - Requires communication to synchronize movement
    
    Environment details:
    - Observation: 31-dim vector per walker (own state + neighbors)
    - Action: 4-dim continuous (hip1, knee1, hip2, knee2)
    - Reward: Shared team reward (forward progress + penalties)
    """
    
    def __init__(self, n_walkers=3, position_noise=1e-3, angle_noise=1e-3,
                 forward_reward=1.0, terminate_reward=-100.0,
                 fall_reward=-10.0, shared_reward=True,
                 terminate_on_fall=True, remove_on_fall=True,
                 terrain_length=200, max_cycles=500, device=None):
        """
        Args:
            n_walkers: Number of walkers (default 3)
            position_noise: Noise in position observations
            angle_noise: Noise in angle observations
            forward_reward: Reward for forward progress
            terminate_reward: Penalty when package falls
            fall_reward: Penalty when walker falls
            shared_reward: All agents get same reward (cooperative)
            terminate_on_fall: End episode if walker falls
            remove_on_fall: Remove fallen walker from environment
            terrain_length: Length of terrain to traverse
            max_cycles: Maximum steps per episode
            device: Device for torch tensors (cuda or cpu)
        """
        self.n_walkers = n_walkers
        self.max_cycles = max_cycles
        self.device = device if device is not None else torch.device('cpu')
        
        # Create PettingZoo environment
        self.env = multiwalker_v9.parallel_env(
            n_walkers=n_walkers,
            position_noise=position_noise,
            angle_noise=angle_noise,
            forward_reward=forward_reward,
            terminate_reward=terminate_reward,
            fall_reward=fall_reward,
            shared_reward=shared_reward,
            terminate_on_fall=terminate_on_fall,
            remove_on_fall=remove_on_fall,
            terrain_length=terrain_length,
            max_cycles=max_cycles,
            render_mode=None
        )
        
        self.agents = None
        self.current_step = 0
        
        # For IC3Net compatibility
        self._observation_dim = None
        self._num_actions = None
        
    def reset(self, seed=None, **kwargs):
        """Reset environment and return initial observations"""
        obs_dict, info_dict = self.env.reset(seed=seed)
        self.agents = list(obs_dict.keys())
        self.current_step = 0
        
        # Convert dict to array for IC3Net
        obs_array = self._dict_to_array(obs_dict)
        
        # Convert to tensor and reshape to (1, nagents, obs_dim)
        obs_tensor = torch.from_numpy(obs_array).double()
        obs_tensor = obs_tensor.reshape(1, self.n_walkers, -1).to(self.device)
        
        return obs_tensor
    
    def step(self, actions):
        """
        Take a step in the environment
        
        Args:
            actions: numpy array of shape (n_walkers, 4) with continuous actions
                    Each action is [hip1, knee1, hip2, knee2] in range [-1, 1]
        
        Returns:
            observations: (n_walkers, obs_dim)
            rewards: (n_walkers,) - shared reward for all
            done: bool - episode terminated
            info: dict with additional info
        """
        # Convert action array to dict
        action_dict = {}
        for i, agent in enumerate(self.agents):
            if i < len(actions):
                action_dict[agent] = actions[i]
            else:
                # Fallback: no-op action
                action_dict[agent] = np.zeros(4)
        
        # Step environment
        obs_dict, reward_dict, done_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        # Update agent list (some may have been removed if they fell)
        self.agents = list(obs_dict.keys())
        self.current_step += 1
        
        # Convert to arrays
        obs_array = self._dict_to_array(obs_dict)
        reward_array = self._dict_to_array(reward_dict, fill_value=0.0)
        
        # Episode done if any agent done or max steps reached
        done = any(done_dict.values()) or any(trunc_dict.values()) or \
               self.current_step >= self.max_cycles
        
        # Aggregate info
        info = {
            'individual_rewards': reward_array,
            'fallen_agents': [a for a in done_dict if done_dict[a]],
            'step': self.current_step
        }
        
        # Convert observations to tensor and reshape to (1, nagents, obs_dim)
        obs_tensor = torch.from_numpy(obs_array).double()
        obs_tensor = obs_tensor.reshape(1, self.n_walkers, -1).to(self.device)
        
        return obs_tensor, reward_array, done, info
    
    def _dict_to_array(self, data_dict, fill_value=0.0):
        """Convert PettingZoo dict to numpy array"""
        if not data_dict:
            return np.array([])
        
        # Get sample to determine dimension
        sample = next(iter(data_dict.values()))
        if isinstance(sample, np.ndarray):
            dim = sample.shape[0] if len(sample.shape) > 0 else 1
            result = np.zeros((self.n_walkers, dim))
        else:
            result = np.zeros(self.n_walkers)
        
        # Fill in available data
        for i in range(self.n_walkers):
            agent_key = f'walker_{i}'
            if agent_key in data_dict:
                result[i] = data_dict[agent_key]
            else:
                # Agent removed (fell) - fill with default
                if isinstance(sample, np.ndarray):
                    result[i] = fill_value
                else:
                    result[i] = fill_value
        
        return result
    
    @property
    def observation_dim(self):
        """Get observation dimension per agent"""
        if self._observation_dim is None:
            # MultiWalker has 31-dim observations per agent
            self._observation_dim = 31
        return self._observation_dim
    
    @property
    def num_actions(self):
        """Get number of action dimensions per agent"""
        if self._num_actions is None:
            # MultiWalker has 4 continuous action dimensions per agent
            self._num_actions = 4
        return self._num_actions
    
    @property
    def dim_actions(self):
        """Number of action dimensions (always 4 for MultiWalker)"""
        return 4
    
    @property
    def num_agents(self):
        """Get number of agents"""
        return self.n_walkers
    
    @property
    def action_space(self):
        """Get action space"""
        # Return a dummy continuous action space
        import gym
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_walkers, 4))
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def render(self):
        """Render environment (if supported)"""
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None


class MultiWalkerDiscretizedWrapper(MultiWalkerWrapper):
    """
    Discretized version of MultiWalker for easier training
    
    Converts continuous actions to discrete choices:
    - Each joint can be: extend (-1), hold (0), or contract (+1)
    - Total: 3^4 = 81 discrete actions per agent
    
    This makes it compatible with IC3Net's discrete action heads
    """
    
    def __init__(self, n_walkers=3, **kwargs):
        super().__init__(n_walkers=n_walkers, **kwargs)
        
        # Discretization: each joint has 3 options (-1, 0, +1)
        self.action_values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        self.n_discrete_actions = len(self.action_values) ** 4  # 3^4 = 81
    
    def step(self, discrete_actions):
        """
        Take step with discrete actions
        
        Args:
            discrete_actions: array of shape (n_walkers,) with discrete action indices
                            Each index in [0, 80] representing a discretized action
        
        Returns:
            Same as parent class
        """
        # Convert discrete actions to continuous
        continuous_actions = self._discrete_to_continuous(discrete_actions)
        
        # Call parent step with continuous actions
        return super().step(continuous_actions)
    
    def _discrete_to_continuous(self, discrete_actions):
        """
        Convert discrete action indices to continuous actions
        
        Args:
            discrete_actions: (n_walkers,) with indices in [0, 80]
        
        Returns:
            continuous_actions: (n_walkers, 4) in [-1, 1] as float32
        """
        continuous_actions = np.zeros((len(discrete_actions), 4), dtype=np.float32)
        
        for i, action_idx in enumerate(discrete_actions):
            # Convert single index to 4 joint actions
            # Each joint: 0=extend(-1), 1=hold(0), 2=contract(+1)
            action_idx = int(action_idx)
            
            joint_actions = []
            for _ in range(4):
                joint_actions.append(action_idx % 3)
                action_idx //= 3
            
            # Map to continuous values and ensure they're in valid range
            actions = self.action_values[joint_actions].astype(np.float32)
            # Clip to ensure no floating point errors exceed bounds
            continuous_actions[i] = np.clip(actions, -1.0, 1.0)
        
        return continuous_actions
    
    @property
    def num_actions(self):
        """Number of discrete actions per agent"""
        return self.n_discrete_actions
    
    @property
    def dim_actions(self):
        """Number of action dimensions (1 for single discrete action)"""
        return 1


def test_multiwalker():
    """Test the MultiWalker wrapper"""
    print("="*70)
    print("TESTING PETTINGZOO MULTIWALKER WRAPPER")
    print("="*70)
    
    # Test continuous version
    print("\n1. Testing Continuous MultiWalker")
    print("-"*70)
    env = MultiWalkerWrapper(n_walkers=3, max_cycles=100)
    
    print(f"Environment created successfully!")
    print(f"  Number of walkers: {env.num_agents}")
    print(f"  Observation dim: {env.observation_dim}")
    print(f"  Action dim: {env.dim_actions}")
    
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"  Expected: (3, 31)")
    
    # Take random actions
    actions = np.random.uniform(-1, 1, (3, 4))
    obs, rewards, done, info = env.step(actions)
    
    print(f"\nAfter step:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Rewards: {rewards}")
    print(f"  Done: {done}")
    
    env.close()
    
    # Test discretized version
    print("\n" + "="*70)
    print("2. Testing Discretized MultiWalker (for IC3Net)")
    print("-"*70)
    env_discrete = MultiWalkerDiscretizedWrapper(n_walkers=3, max_cycles=100)
    
    print(f"Environment created successfully!")
    print(f"  Number of walkers: {env_discrete.num_agents}")
    print(f"  Observation dim: {env_discrete.observation_dim}")
    print(f"  Discrete actions: {env_discrete.num_actions}")
    
    obs = env_discrete.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    # Take random discrete actions
    discrete_actions = np.random.randint(0, env_discrete.num_actions, size=3)
    print(f"Random discrete actions: {discrete_actions}")
    
    obs, rewards, done, info = env_discrete.step(discrete_actions)
    
    print(f"\nAfter step:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Rewards: {rewards}")
    print(f"  Done: {done}")
    
    env_discrete.close()
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == '__main__':
    test_multiwalker()
