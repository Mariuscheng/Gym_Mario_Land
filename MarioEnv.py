import matplotlib
from pyboy import PyBoy

import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

from enum import Enum
from collections import deque, defaultdict, namedtuple
import numpy as np
import random
import math
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# set up matplotlib
#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#    from IPython import display

#plt.ion()

class Actions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    RUN = 5
    FIRE = 6
    JUMP = 7
    LONG_JUMP = 8
    
class MarioEnv(gym.Env):
    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 20), dtype=np.uint16)
        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == Actions.NOOP.value:
            pass
        elif action == Actions.LEFT.value:
            self.pyboy.button("left")
        elif action == Actions.RIGHT.value:
            self.pyboy.button("right")
        elif action == Actions.UP.value:
            self.pyboy.button("up")
        elif action == Actions.DOWN.value:
            self.pyboy.button("down")
        elif action == Actions.RUN.value:
            self.pyboy.button_press("b")
            self.pyboy.button("right")
        elif action == Actions.FIRE.value:
            self.pyboy.button("b")
        elif action == Actions.JUMP.value:
            self.pyboy.button_press("a")
        elif action == Actions.LONG_JUMP.value:
            self.pyboy.button_press("b")
            self.pyboy.button_press("right")
            self.pyboy.button("a")

        self.pyboy.tick()
        
        # Check if game is over (terminated) or if we need to truncate the episode
        terminated = bool(self.pyboy.game_wrapper.game_over())
        truncated = False  # We can set custom truncation conditions here if needed
        
        self._calculate_fitness()
        reward = self._fitness-self._previous_fitness
        
        observation=self._get_obs()
        info = self.pyboy.game_wrapper
        return observation, reward, terminated, truncated, info
    
    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        
        # Get current progress in the level
        current_progress = self.pyboy.game_wrapper.level_progress
        
        # Base fitness is the level progress (how far right Mario has moved)
        progress_reward = current_progress * 0.1
        
        # Add score component (if available)
        score_component = self.pyboy.game_wrapper.score if hasattr(self.pyboy.game_wrapper, 'score') else 0
        
        # Penalize for losing lives
        lives_penalty = -50 if self.pyboy.game_wrapper.lives_left == 0 else 0
        
        # Calculate total fitness
        self._fitness = progress_reward + score_component * 0.01 + lives_penalty
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._fitness = 0
        self._previous_fitness = 0
            
        observation = self._get_obs()
        info = {}
        return observation, info

    def render(self):
        self.pyboy.tick()

    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
        self.pyboy.game_wrapper.game_area_mapping(self.pyboy.game_wrapper.mapping_compressed, 0)
        return self.pyboy.game_area()

# pyboy = PyBoy("rom.gb", window="SDL2")
# env = MarioEnv(pyboy)

#print(env.action_space) # 8
#print(env.observation_space.shape) # (16, 20)

#mario = pyboy.game_wrapper

#assert mario.lives_left == 2

# observation , info = env.reset()

# n_actions = env.action_space.n
# n_observations = len(torch.tensor((observation)))

#print(n_observations) #320
# tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0, 14, 14,  0,  1,  1,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0, 14, 14,  0,  1,  1,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 10, 10, 10, 10, 10, 10, 10,
#         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
#         10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
#        dtype=torch.uint16)
# 320