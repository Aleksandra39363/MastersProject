#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math

# 3rd party modules
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ic3net_envs.traffic_helper import *

# Optional: curses is only used for terminal rendering and is unavailable on Windows
try:
    import curses
except ModuleNotFoundError:
    curses = None


def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class TrafficJunctionEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # Minimal spaces so gymnasium's PassiveEnvChecker passes on make()
        try:
            self.action_space = spaces.Discrete(1)
            # placeholder observation space; will be overwritten in multi_agent_init
            self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        except Exception:
            pass

        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.0001  # Weak step penalty to allow agents to learn avoiding crashes
        self.CRASH_PENALTY = -2.0  # Base crash penalty (overridden by --crash_penalty flag)
        self.TERMINAL_REWARD = 10.0  # Base completion reward (overridden by --terminal_reward flag)
        self.SUCCESS_BONUS = 0.0  # Disabled: use --terminal_reward to control completion reward
        self.PROGRESS_REWARD = 0.05  # Small reward for making progress along route

        self.episode_over = False
        self.has_failed = 0

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Traffic Junction task')
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box (i.e length of road) ")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of car")
        env.add_argument('--add_rate_min', type=float, default=0.05,
                         help="rate at which to add car (till curr. start)")
        env.add_argument('--add_rate_max', type=float, default=0.2,
                         help=" max rate at which to add car")
        env.add_argument('--curr_start', type=float, default=0,
                         help="start making harder after this many epochs [0]")
        env.add_argument('--curr_end', type=float, default=0,
                         help="when to make the game hardest [0]")
        env.add_argument('--crash_penalty', type=float, default=-2.0,
                 help="penalty applied to each agent on collision")
        env.add_argument('--terminal_reward', type=float, default=15.0,
                 help="reward applied when an agent reaches destination")
        env.add_argument('--difficulty', type=str, default='easy',
                         help="Difficulty level, easy|medium|hard")
        env.add_argument('--vocab_type', type=str, default='bool',
                         help="Type of location vector to use, bool|scalar")


    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'add_rate_min', 'add_rate_max', 'curr_start', 'curr_end',
              'crash_penalty', 'terminal_reward',
                  'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))

        self.CRASH_PENALTY = self.crash_penalty
        self.TERMINAL_REWARD = self.terminal_reward

        # Respect pre-set ncar (from data.py) if it exists, otherwise use nfriendly or nagents
        if not hasattr(self, 'ncar') or self.ncar is None:
            self.ncar = getattr(args, 'nfriendly', None) or getattr(args, 'nagents', 1)
        self.dims = dims = (self.dim, self.dim)
        difficulty = args.difficulty
        vision = args.vision

        if difficulty in ['medium','easy']:
            assert dims[0]%2 == 0, 'Only even dimension supported for now.'

            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision'

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.naction = 2
        self.action_space = spaces.Discrete(self.naction)

        # make no. of dims odd for easy case.
        if difficulty == 'easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            # Observation space is a Tuple of per-agent observations
            agent_obs_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            self.observation_space = spaces.Tuple([agent_obs_space] * self.ncar)
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (act, route_id, position, vision_square)
            agent_obs_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiDiscrete(dims),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            # observation_space is Tuple of per-agent observations (ncar agents total)
            self.observation_space = spaces.Tuple([agent_obs_space] * self.ncar)

        self._set_grid()

        if difficulty == 'easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

        return

    def reset(self, epoch=None, *args, seed=None, options=None, **kwargs):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        # Accept seed and options from gymnasium API
        if seed is not None:
            np.random.seed(seed)
        
        self.episode_over = False
        self.has_failed = 0
        self.episode_total_reward = 0.0  # Track total reward for episode
        self.episode_completions = 0  # Track total completions for episode
        self.episode_crashes = np.zeros(self.ncar)  # Track crashes per agent
        self.total_agents_spawned = 0  # Track total agents ever spawned (not currently in system)
        self.last_episode_stat = None  # Store stats from last completed episode

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)

        # stat - like success ratio
        self.stat = dict()

        # Reset episode flags for new episode
        self.episode_over = False
        self.has_failed = 0
        self.is_completed = np.zeros(self.ncar)

        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch
        
        # Set stat['add_rate'] AFTER curriculum update
        self.stat['add_rate'] = self.add_rate

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, terminated, truncated, info : tuple
            obs (object) : observation
            reward (ndarray) : reward array for all agents
            terminated (bool) : episode terminated
            truncated (bool) : episode truncated by time limit
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either ncar or ncar x 1
        action = np.array(action)
        # Only squeeze the second dimension (for ncar x 1 -> ncar), not all dimensions
        if action.ndim == 2:
            action = action.squeeze(axis=1)
        # Ensure action is at least 1-D (don't squeeze away all dimensions)
        action = np.atleast_1d(action)

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."
        
        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            self._take_action(i, a)

        self._add_cars()

        obs = self._get_obs()
        rewards = self._get_reward()
        
        # Add terminal rewards immediately when agents complete
        # Must happen here because is_completed is reset to zeros at start of next step
        # Each agent rewarded independently for their own completion
        terminal_rewards = self.is_completed.astype(float) * self.TERMINAL_REWARD  # Per-agent reward
        rewards = rewards + terminal_rewards

        # Track episode completion count
        self.episode_completions += int(np.sum(self.is_completed))

        debug = {'car_loc':self.car_loc,
            'alive_mask': np.copy(self.alive_mask),
            'wait': self.wait,
            'cars_in_sys': self.cars_in_sys,
            'is_completed': np.copy(self.is_completed)}

        # Update stats (capture current episode state)
        self.stat['add_rate'] = self.add_rate
        
        # Success rate = binary metric per episode
        # 1.0 if ALL agents completed their routes AND no crashes occurred
        # 0.0 otherwise
        all_completed = (self.episode_completions == self.ncar)
        no_crashes = (np.sum(self.episode_crashes) == 0)
        self.stat['success'] = 1.0 if (all_completed and no_crashes) else 0.0
        
        # Track additional metrics for analysis
        self.stat['completion_rate'] = self.episode_completions / self.ncar if self.ncar > 0 else 0
        self.stat['total_crashes'] = int(np.sum(self.episode_crashes))

        # Success bonus: reward all agents only when the whole episode is fully successful
        if self.cars_in_sys == 0 and self.stat['success'] == 1.0:
            rewards = rewards + self.SUCCESS_BONUS
            debug['success_bonus'] = self.SUCCESS_BONUS
        else:
            debug['success_bonus'] = 0.0

        # Track episode reward after all reward shaping is applied
        self.episode_total_reward += np.sum(rewards)
        
        # Crashes don't end episode - they just remove individual agents
        # Episode terminates only when all agents are removed (cars_in_sys == 0)
        if self.cars_in_sys == 0:
            self.episode_over = True
            # Store stats when episode truly ends
            self.last_episode_stat = {
                'success': self.stat['success'],
                'completion_rate': self.stat['completion_rate'],
                'total_crashes': self.stat['total_crashes'],
                'add_rate': self.stat['add_rate']
            }
            # Reset episode trackers for next episode
            self.episode_completions = 0
            self.episode_crashes = np.zeros(self.ncar)

        terminated = self.episode_over
        truncated = False

        # Return scalar reward for gymnasium compatibility
        debug['rewards'] = rewards
        return obs, float(np.sum(rewards)), terminated, truncated, debug

    def render(self, mode='human', close=False):

        grid = self.grid.copy().astype(object)
        # grid = np.zeros(self.dims[0]*self.dims[1], dtypeobject).reshape(self.dims)
        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.stdscr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0: # GAS
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<>'
                else:
                    grid[p[0]][p[1]] = '<>'
            else: # BRAKE
                if grid[p[0]][p[1]] != 0:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<b>'
                else:
                    grid[p[0]][p[1]] = '<b>'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:
                    continue
                if item != '_':
                    if '<>' in item and len(item) > 3: #CRASH, one car accelerates
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif '<>' in item: #GAS
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1))
                    elif 'b' in item and len(item) > 3: #CRASH
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2))
                    elif 'b' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(5))
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2))
                else:
                    self.stdscr.addstr(row_num, idx * 4, '_'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self):
        return

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty)
        for road in roads:
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]


        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.naction - 1)

            # route id
            r_i = self.route_id[i] / (self.npath - 1)

            # loc
            p_norm = p / (h-1, w-1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)

        return obs


    def _add_cars(self):
        for r_i, routes in enumerate(self.routes):
            # Stop spawning once we've spawned ncar agents total (even if they've completed)
            # This ensures episodes naturally end when all agents have passed through
            if self.total_agents_spawned >= self.ncar:
                return

            # Add car to system and set on path
            if np.random.uniform() <= self.add_rate:

                # chose dead car on random
                idx = self._choose_dead()
                
                # choose path randomly & set it
                p_i = np.random.choice(len(routes))
                start_pos = routes[p_i][0]
                
                # Check if start position is occupied by another alive car
                position_occupied = False
                for other_idx in range(self.ncar):
                    if other_idx != idx and self.alive_mask[other_idx] == 1:
                        if np.array_equal(self.car_loc[other_idx], start_pos):
                            position_occupied = True
                            break
                
                # Only spawn if position is free
                if not position_occupied:
                    # make it alive
                    self.alive_mask[idx] = 1
                    
                    # make sure all self.routes have equal len/ same no. of routes
                    self.route_id[idx] = p_i + r_i * len(routes)
                    self.chosen_path[idx] = routes[p_i]

                    # set its start loc
                    self.car_route_loc[idx] = 0
                    self.car_loc[idx] = start_pos

                    # increase count
                    self.cars_in_sys += 1
                    self.total_agents_spawned += 1  # Track total spawned across entire episode

    def _set_paths_easy(self):
        h, w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())


    def _set_paths_medium_old(self):
        h,w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': [],
            'RIGHT': [],
            'DOWN': []
        }

        # type 0 paths: go straight on junction
        # type 1 paths: take right on junction
        # type 2 paths: take left on junction


        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2-1) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to UP to LEFT, type 1
        first_half = full[:h//2]
        second_half = [(h//2 - 1, i) for i in range(w//2 - 2,-1,-1) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))

        # 2 refers to UP to RIGHT, type 2
        second_half = [(h//2, i) for i in range(w//2-1, w) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))


        # 3 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        # 4 refers to LEFT to DOWN, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2 - 1) for i in range(h//2+1, h)]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))

        # 5 refers to LEFT to UP, type 2
        second_half = [(i, w//2) for i in range(h//2, -1,-1) ]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))


        # 6 refers to DOWN to UP, type 0
        full = [(i, w//2) for i in range(h-1,-1,-1)]
        self.routes['DOWN'].append(np.array([*full]))

        # 7 refers to DOWN to RIGHT, type 1
        first_half = full[:h//2]
        second_half = [(h//2, i) for i in range(w//2+1,w)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))

        # 8 refers to DOWN to LEFT, type 2
        second_half = [(h//2-1, i) for i in range(w//2,-1,-1)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))


        # 9 refers to RIGHT to LEFT, type 0
        full = [(h//2-1, i) for i in range(w-1,-1,-1)]
        self.routes['RIGHT'].append(np.array([*full]))

        # 10 refers to RIGHT to UP, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2) for i in range(h//2 -2, -1,-1)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))

        # 11 refers to RIGHT to DOWN, type 2
        second_half = [(i, w//2-1) for i in range(h//2-1, h)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))


        # PATHS_i: 0 to 11
        # 0 refers to UP to down,
        # 1 refers to UP to left,
        # 2 refers to UP to right,
        # 3 refers to LEFT to right,
        # 4 refers to LEFT to down,
        # 5 refers to LEFT to up,
        # 6 refers to DOWN to up,
        # 7 refers to DOWN to right,
        # 8 refers to DOWN to left,
        # 9 refers to RIGHT to left,
        # 10 refers to RIGHT to up,
        # 11 refers to RIGHT to down,

        # Convert to routes dict to list of paths
        paths = []
        for r in self.routes.values():
            for p in r:
                paths.append(p)

        # Check number of paths
        # assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
        self.routes = get_routes(self.dims, route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)


    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True


    def _take_action(self, idx, act):
        # non-active car
        if self.alive_mask[idx] == 0:
            return

        # add wait time for active cars
        self.wait[idx] += 1

        # action BRAKE i.e STAY
        if act == 1:
            self.car_last_act[idx] = 1
            return

        # GAS or move
        if act==0:
            prev_loc = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]
            
            # Small progress reward for advancing along route (commented out by default)
            # Uncomment to enable reward shaping for easier learning
            # if curr < len(self.chosen_path[idx]) and curr > prev_loc:
            #     self.progress_rewards[idx] = self.PROGRESS_REWARD
            
            prev = prev_loc

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of boud car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0



    def _get_reward(self):
        reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait

        for i, l in enumerate(self.car_loc):
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or \
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any():
               reward[i] += self.CRASH_PENALTY
               self.episode_crashes[i] += 1  # Track crashes per agent for efficient success
               self.has_failed = 1
               # Remove crashed car from episode (prevent it from continuing)
               if self.alive_mask[i] == 1:
                   self.alive_mask[i] = 0  
                   self.cars_in_sys -= 1

        reward = self.alive_mask * reward
        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = list(np.ogrid[tuple(map(slice, idx.shape))])
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        """Give positive reward for cars that successfully completed their routes
        
        Each agent receives independent reward for their own completion.
        No shared/team reward - IC3Net learns per-agent policies.
        """
        # Zero rewards because terminal rewards already given in step()
        # This method kept for compatibility
        return np.zeros(self.ncar)

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)
