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
    # Direction pad values
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    
    # Function button values (A and B buttons)
    BUTTON_NOOP = 0
    BUTTON_A = 1  # Jump
    BUTTON_B = 2  # Run
    BUTTON_AB = 3  # Jump while running

# Create action groups for easier reference
DIRECTION_ACTIONS = [Actions.NOOP, Actions.LEFT, Actions.RIGHT, Actions.UP, Actions.DOWN]
FUNCTION_ACTIONS = [Actions.BUTTON_NOOP, Actions.BUTTON_A, Actions.BUTTON_B, Actions.BUTTON_AB]

class MarioEnv(gym.Env):
    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.pyboy = pyboy
        self.debug = debug

        # Define action space using the number of actions in each group
        self.n_directions = len(DIRECTION_ACTIONS)
        self.n_functions = len(FUNCTION_ACTIONS)
        self.action_space = spaces.MultiDiscrete([self.n_directions, self.n_functions])
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 20), dtype=np.uint16)
        self.pyboy.game_wrapper.start_game()

    def _convert_action(self, action_int):
        """Convert single integer action to multi-discrete format."""
        # Convert single integer to direction and function indices
        direction_idx = action_int // self.n_functions
        function_idx = action_int % self.n_functions
        
        #if self.debug:
            #print(f"Action {action_int} -> Direction: {DIRECTION_ACTIONS[direction_idx].name}, Function: {FUNCTION_ACTIONS[function_idx].name}")
        
        return np.array([direction_idx, function_idx])

    def step(self, action):
        # Convert single integer action to multi-discrete format if needed
        if isinstance(action, (int, np.integer)):
            action = self._convert_action(action)
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Handle direction pad first (always release previous buttons)
        self.pyboy.button_release("left")
        self.pyboy.button_release("right")
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        
        if action[0] == Actions.LEFT.value:
            self.pyboy.button_press("left")
        elif action[0] == Actions.RIGHT.value:
            self.pyboy.button_press("right")
        elif action[0] == Actions.UP.value:
            self.pyboy.button_press("up")
        elif action[0] == Actions.DOWN.value:
            self.pyboy.button_press("down")

        # Handle function buttons (A and B) - always release previous buttons
        self.pyboy.button_release("a")
        self.pyboy.button_release("b")
        
        if action[1] == Actions.BUTTON_A.value:  # Jump
            self.pyboy.button_press("a")
        elif action[1] == Actions.BUTTON_B.value:  # Run
            self.pyboy.button_press("b")
        elif action[1] == Actions.BUTTON_AB.value:  # Jump while running
            self.pyboy.button_press("b")  # Press B first for running
            self.pyboy.button_press("a")  # Then jump
            
        # Always tick after all buttons are pressed
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
        super().reset(seed=40)
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

#pyboy = PyBoy("rom.gb", window="null")
#env = MarioEnv(pyboy)

#print(env.action_space) #MultiDiscrete([5 4])
#print(env.observation_space.shape) #(16, 20)
#print(env.action_space.sample()) #[1 3]

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
