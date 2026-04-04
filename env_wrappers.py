import time
import numpy as np
import torch
from gymnasium import spaces
from inspect import getfullargspec as getargspec
# from inspect import getargspec

class GymWrapper(object):
    '''
    for multi-agent
    '''
    def __init__(self, env, device=None):
        self.env = env
        self.device = device if device is not None else torch.device('cpu')

    @property
    def num_agents(self):
        '''Get number of agents from the underlying environment'''
        # Unwrap any gymnasium wrappers to get to the actual custom env
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        
        # Check common agent count attributes
        if hasattr(env, 'n_agents'):
            return env.n_agents
        elif hasattr(env, 'nagents'):
            return env.nagents
        elif hasattr(env, 'ncar'):
            return env.ncar
        elif hasattr(env, 'npredator'):
            return env.npredator
        else:
            # Fallback: assume 1 agent if not found
            return 1

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # Get the first agent's observation space (they're all the same)
        if hasattr(self.env.observation_space, 'spaces'):
            # observation_space = Tuple([agent_obs_space] * n_agents)
            # We only care about ONE agent's obs space
            agent_obs_space = self.env.observation_space.spaces[0]
            
            # Now calculate the size of this single agent's observation
            if hasattr(agent_obs_space, 'spaces'):
                # agent_obs_space = Tuple((action, route_id, vision_grid, ...))
                # Sum up all components
                total_obs_dim = 0
                for component_space in agent_obs_space.spaces:
                    if hasattr(component_space, 'shape') and component_space.shape is not None:
                        total_obs_dim += int(np.prod(component_space.shape))
                    elif hasattr(component_space, 'n'):
                        # Discrete space
                        total_obs_dim += 1
                    else:
                        # Fallback for other space types
                        total_obs_dim += 1
                return total_obs_dim
            else:
                # Single agent obs is not a Tuple
                if hasattr(agent_obs_space, 'shape') and agent_obs_space.shape is not None:
                    return int(np.prod(agent_obs_space.shape))
                elif hasattr(agent_obs_space, 'n'):
                    return 1
                else:
                    return 1
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, epoch=None):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            result = self.env.reset(epoch)
        else:
            result = self.env.reset()

        # gym / gymnasium compatibility: reset may return obs or (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        obs = self._flatten_obs(obs)
        return obs

    def display(self):
        self.env.render()
        time.sleep(0.5)

    def end_display(self):
        self.env.exit_render()

    def step(self, action):
        # action is already an array from action_utils.translate_action
        # The multi-agent environments expect array-like actions, so pass through directly
        result = self.env.step(action)
        # Support both gym (obs, reward, done, info) and gymnasium (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple):
            if len(result) == 5:
                obs, r, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                obs, r, done, info = result
                # normalize to variables used below
                terminated = done
                truncated = False
            else:
                raise RuntimeError("Unsupported env.step() return signature: {}".format(len(result)))
        else:
            raise RuntimeError("env.step() did not return a tuple")

        # If the core env returned a scalar reward but provided per-agent
        # rewards in info['rewards'], prefer passing the per-agent array
        # forward so legacy trainer code that expects per-agent rewards keeps working.
        if not (isinstance(r, (float, int, np.floating, np.integer))) and 'rewards' in info:
            # prefer explicit info field
            r = np.array(info.get('rewards'))
        elif (isinstance(r, (float, int, np.floating, np.integer))) and 'rewards' in info:
            # the env returned scalar for gymnasium compatibility but included per-agent rewards
            r = np.array(info.get('rewards'))

        obs = self._flatten_obs(obs)
        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)

        # obs shape after stack: (nagents, obs_dim_per_agent)
        # reshape to (batch_size=1, nagents, obs_dim_per_agent)
        nagents = self.num_agents
        obs = obs.reshape(1, nagents, self.observation_dim)
        obs = torch.from_numpy(obs).double().to(self.device)
        return obs

    def get_stat(self):
        # Unwrap any gymnasium wrappers to get to the actual custom env with stat
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        
        if hasattr(env, 'last_episode_stat') and env.last_episode_stat is not None:
            # Return only the last completed episode's stats (not accumulated)
            return env.last_episode_stat
        elif hasattr(env, 'stat'):
            # Fallback to current stat dict if no last_episode_stat
            env.stat.pop('steps_taken', None)
            return env.stat
        else:
            return dict()

    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
