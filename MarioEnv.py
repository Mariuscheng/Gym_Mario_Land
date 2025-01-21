from pyboy import PyBoy
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Discrete
from gymnasium import spaces
from enum import Enum
import numpy as np

import cv2

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Actions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    JUMP = 5
    FIRE = 6
    LEFT_PRESS = 7
    RIGHT_PRESS = 8
    JUMP_PRESS = 9
    
# class ARROW_Function(Enum):
#     NOOP = 0
#     LEFT = 1
#     RIGHT = 2
#     UP = 3
#     DOWN = 4
#     LEFT_PRESS = 5
#     RIGHT_PRESS = 6
    
# class A_FUNCTION(Enum):
#     NOOP = 0
#     BUTTON_A = 1  # Jump

# class B_FUNCTION(Enum):
#     NOOP = 0
#     BUTTON_B = 1  # RUN OR FIRE

# Create action groups for easier reference
# DIRECTION_ACTIONS = [ARROW_Function.NOOP, ARROW_Function.LEFT, ARROW_Function.RIGHT, ARROW_Function.UP, ARROW_Function.DOWN, ARROW_Function.LEFT_PRESS, ARROW_Function.RIGHT_PRESS]
# A_FUNCTION_ACTIONS = [A_FUNCTION.NOOP, A_FUNCTION.BUTTON_A]
# B_FUNCTION_ACTIONS = [B_FUNCTION.NOOP, B_FUNCTION.BUTTON_B]

class MarioEnv(gym.Env, ):
    def __init__(self, pyboy):
        super().__init__()
        self.pyboy = pyboy
        #self.device = device

        # Define action space using the number of actions in each group
        # self.n_directions = len(DIRECTION_ACTIONS)
        # self.a_functions = len(A_FUNCTION_ACTIONS)
        # self.b_functions = len(B_FUNCTION_ACTIONS)
        # self.action_space = MultiDiscrete(np.array([self.n_directions, self.a_functions, self.b_functions]), seed=42)
        self.action_space = Discrete(len(Actions), seed=42)

        # Define observation space
        self.observation_space = Box(low=0, high=255, shape=(16, 20), dtype=np.uint8)  # 假設距離的值範圍是 0-255
        
        # Initialize fitness attributes
        self._fitness = 0
        self._previous_fitness = 0
        
        #self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        # Move the agent
        if action == Actions.NOOP.value:
            pass
        elif action == Actions.LEFT_PRESS.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_PRESS.value:
            self.pyboy.button_press("right")
        elif action == Actions.LEFT.value:
            self.pyboy.button("left")
        elif action == Actions.RIGHT.value:
            self.pyboy.button("right")
        elif action == Actions.UP.value:
            self.pyboy.button("up")
        # elif action == Actions.DOWN.value:
        #     self.pyboy.button("down")   
        elif action == Actions.JUMP_PRESS.value:
            self.pyboy.button_press("b")
            self.pyboy.button_press("a")
            self.pyboy.button_press("right")
        elif action == Actions.JUMP.value:
            self.pyboy.button("a")   
        elif action == Actions.FIRE.value:
            self.pyboy.button("b")

        # # Move the agent
        # if action == ARROW_Function[0].NOOP.value:
        #     pass
        # elif action == ARROW_Function.LEFT_PRESS.value:
        #     self.pyboy.button_press("left")
        # elif action == ARROW_Function.RIGHT_PRESS.value:
        #     self.pyboy.button_press("right")
        # elif action == ARROW_Function.LEFT.value:
        #     self.pyboy.button("left")
        # elif action == ARROW_Function.RIGHT.value:
        #     self.pyboy.button("right")
        # elif action == ARROW_Function.UP.value:
        #     self.pyboy.button("up")
        # elif action == ARROW_Function.DOWN.value:
        #     self.pyboy.button("down")
            
        # # A button actions for the agent    
        # if action == A_FUNCTION.NOOP.value:
        #     pass
        # elif action == A_FUNCTION.BUTTON_A.value:
        #     self.pyboy.button_press("a")
        
        # # B button actions for the agent    
        # if action == B_FUNCTION.NOOP.value:
        #     pass
        # elif action == B_FUNCTION.BUTTON_B.value:
        #     self.pyboy.button("b")
            
        self.pyboy.tick()
        
        # 基於 level_progress 和 lives_left 計算分數
        # if self.mario.level_progress > 251:  # 確保 level_progress 大於 251 才計算增量
        #     level_progress_score = 10
        # else:
        #     level_progress_score = 0
            
        # level_progress_score = max(level_progress_score, self.mario.level_progress)

        # lives_penalty = (2 - self.mario.lives_left) * 100  # 每失去一條命，扣 100 分
        
        # Mario_X_position = self.pyboy.memory[0XC202]
        # right_walk = Mario_X_position + 10
        
        # reward = self.mario.score + level_progress_score - lives_penalty + right_walk
        
        # # Check if game is over (terminated) or if we need to truncate the episode
        # terminated = bool(self.mario.game_over())
        # truncated = self.mario.level_progress >= 2601  # Example threshold for level completion
                
        terminated = self.pyboy.game_wrapper.game_over
        
        reward=self.pyboy.game_wrapper.score

        observation=self.pyboy.game_area()
        
        info = {}
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness=0
        self._previous_fitness=0

        observation=self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self):
        self.pyboy.tick()

    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
         return self.pyboy.game_area()
        
Test Env:
=======================================================
# 初始化 Mario 環境
# from pyboy.api.memory_scanner import DynamicComparisonType        
# pyboy = PyBoy("rom.gb", window="SDL2")    

# env = MarioEnv(pyboy)

# pyboy.game_wrapper.set_world_level(1, 2)
# pyboy.game_wrapper.start_game()

# while pyboy.tick():
#     print(pyboy.game_wrapper)
#     pass
# pyboy.stop()

# observation_size = np.array(env._get_obs())

# print(np.atleast_2d([[0,1],
#                     [16,17]]))

# goomba = 144

# while pyboy.tick():
#     print(pyboy.memory_scanner.scan_memory(goomba, start_addr=0xD100, end_addr=0xD17E))
#     addresses = pyboy.memory_scanner.rescan_memory(goomba, DynamicComparisonType.MATCH)
#     print(addresses)
#     pass
# pyboy.stop()
# s = pyboy.get_sprite(15)

#print(s, s.shape, s.image().save('tile_1.png'))
# print(print(pyboy.get_sprite_by_tile_identifier([144])))

# print(pyboy.memory[0xD100:0xD17E])

# observation, info = env.reset()
# while pyboy.tick():
#     print(pyboy.game_wrapper)
#     pass
# pyboy.stop()
# print(observation)

# print(observation.item(144))

# mario = pyboy.game_wrapper
# mario.game_area_mapping(mario.mapping_compressed, 0)

# while pyboy.tick():
#     print(pyboy.game_wrapper)
#     pass

