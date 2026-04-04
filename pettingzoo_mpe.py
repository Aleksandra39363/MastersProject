"""
PettingZoo MPE (Multi-Particle Environment) Integration for IC3Net
Environment: Simple Spread - Cooperative landmark coverage task
"""

import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3


class MPESimpleSpreadWrapper:
    """
    Wrapper for PettingZoo's Simple Spread environment
    
    Task: N agents must cover N landmarks while avoiding collisions
    Observation: Position, velocity, landmark positions, other agents
    Action: Discrete (5 actions: no-op, left, right, up, down)
    Reward: Shared reward based on landmark coverage + collision penalty
    """
    
    def __init__(self, n_agents=3, max_cycles=25, continuous_actions=False, device=None):
        """
        Args:
            n_agents: Number of agents (same as number of landmarks)
            max_cycles: Maximum steps per episode
            continuous_actions: Whether to use continuous actions (False = discrete 5 actions)
            device: Device for torch tensors (cuda or cpu)
        """
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.device = device if device is not None else torch.device('cpu')
        
        # Create PettingZoo environment
        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions
        )
        
        # Reset to initialize
        self.env.reset()
        
        # Get action and observation dimensions
        agent_name = self.env.agents[0]
        obs_space = self.env.observation_space(agent_name)
        act_space = self.env.action_space(agent_name)
        
        self.obs_size = obs_space.shape[0]
        self.observation_dim = self.obs_size  # Alias for IC3Net compatibility
        
        if continuous_actions:
            self.n_actions = act_space.shape[0]  # Continuous action dims
            self.action_type = 'continuous'
        else:
            self.n_actions = act_space.n  # Discrete actions (5)
            self.action_type = 'discrete'
        
        # All possible aliases for IC3Net compatibility
        self.naction = self.n_actions
        self.num_actions = [self.n_actions]  # Must be a list for IC3Net
        self.dim_actions = 1  # Single action dimension (not multi-headed)
        self.episode_step = 0
        
        # Track statistics
        self.stat = {}
        self.episode_collisions = 0
        self.episode_min_dist = float('inf')
        self.episode_avg_dist = 0.0
        
    def reset(self):
        """Reset environment and return initial observations"""
        # PettingZoo v3 parallel API: reset returns (observations, infos)
        obs_dict, info_dict = self.env.reset()
        self.episode_step = 0
        
        # Reset statistics
        self.episode_collisions = 0
        self.episode_min_dist = float('inf')
        self.episode_avg_dist = 0.0
        self.stat = {}
        
        # Convert dict to array [n_agents, obs_size]
        obs_array = np.array([obs_dict[agent] for agent in self.env.agents])
        
        # Convert to tensor and reshape to (1, nagents, obs_dim)
        obs_tensor = torch.from_numpy(obs_array).double()
        obs_tensor = obs_tensor.reshape(1, self.n_agents, -1).to(self.device)
        
        return obs_tensor
    
    def step(self, actions):
        """
        Take a step in the environment
        
        Args:
            actions: numpy array of shape [n_agents] with action indices
        
        Returns:
            observations: [n_agents, obs_size]
            rewards: [n_agents]
            done: bool
            info: dict
        """
        # Convert actions array to dict
        action_dict = {
            agent: int(actions[i]) for i, agent in enumerate(self.env.agents)
        }
        
        # Step environment (PettingZoo v3 parallel API returns 5 values)
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)
        
        self.episode_step += 1
        
        # Convert dicts to arrays
        obs_array = np.array([obs_dict.get(agent, np.zeros(self.obs_size)) 
                              for agent in self.env.agents])
        reward_array = np.array([reward_dict.get(agent, 0.0) 
                                 for agent in self.env.agents], dtype=np.float64)
        
        # Ensure reward_array has correct shape (n_agents,)
        if reward_array.shape[0] != self.n_agents:
            # Pad or truncate to match expected agent count
            new_reward = np.zeros(self.n_agents, dtype=np.float64)
            new_reward[:min(len(reward_array), self.n_agents)] = reward_array[:min(len(reward_array), self.n_agents)]
            reward_array = new_reward
        
        # Episode done if any agent terminated or truncated
        done = any(term_dict.values()) or any(trunc_dict.values())
        
        # Track distances to landmarks (from observation structure)
        # Simple Spread obs: [self_vel(2), self_pos(2), landmark_pos(2*N), other_pos(2*(N-1))]
        # Landmarks start at index 4, with 2 coords each
        landmark_dists = []
        for agent_obs in obs_array:
            # Extract landmark positions (skip self_vel and self_pos)
            agent_pos = agent_obs[2:4]  # Agent's position
            for i in range(self.n_agents):
                landmark_pos = agent_obs[4 + i*2:4 + (i+1)*2]
                dist = np.linalg.norm(agent_pos - landmark_pos)
                landmark_dists.append(dist)
        
        if len(landmark_dists) > 0:
            min_dist = min(landmark_dists)
            avg_dist = np.mean(landmark_dists)
            self.episode_min_dist = min(self.episode_min_dist, min_dist)
            self.episode_avg_dist = avg_dist
        
        # Update statistics for episode end
        if done:
            self.stat['min_dist'] = self.episode_min_dist
            self.stat['avg_dist'] = self.episode_avg_dist
            self.stat['steps_taken'] = self.episode_step
        
        # Additional info
        info = {
            'episode_step': self.episode_step,
            'mean_reward': np.mean(reward_array),
            'total_reward': np.sum(reward_array),
            'min_landmark_dist': self.episode_min_dist,
            'avg_landmark_dist': self.episode_avg_dist
        }
        
        # Convert observations to tensor and reshape to (1, nagents, obs_dim)
        obs_tensor = torch.from_numpy(obs_array).double()
        obs_tensor = obs_tensor.reshape(1, self.n_agents, -1).to(self.device)
        
        return obs_tensor, reward_array, done, info
    
    def get_stat(self):
        """Get statistics (for compatibility with IC3Net trainer)"""
        # Return episode statistics for MPE Simple Spread
        # Key metrics: min_dist (coverage), avg_dist (efficiency), steps_taken
        return self.stat
    
    def reward_terminal(self):
        """Return terminal rewards (MPE doesn't use terminal rewards, so return zeros)"""
        return np.zeros(self.n_agents)
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment"""
        self.env.close()


def test_environment():
    """Test the MPE wrapper"""
    print("Testing MPE Simple Spread Wrapper...")
    
    env = MPESimpleSpreadWrapper(n_agents=3, max_cycles=25)
    
    print(f"Number of agents: {env.n_agents}")
    print(f"Observation size: {env.obs_size}")
    print(f"Number of actions: {env.n_actions}")
    print(f"Action type: {env.action_type}")
    
    # Test random episode
    obs = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(25):
        # Random actions
        actions = np.random.randint(0, env.n_actions, size=env.n_agents)
        obs, rewards, done, info = env.step(actions)
        
        total_reward += np.sum(rewards)
        
        if step % 5 == 0:
            print(f"Step {step}: Rewards = {rewards}, Done = {done}")
        
        if done:
            print(f"Episode ended at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Test completed successfully!")
    
    env.close()


if __name__ == '__main__':
    test_environment()
