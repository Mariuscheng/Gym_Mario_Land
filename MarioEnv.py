import gymnasium as gym
from gymnasium import spaces

import numpy as np
from pyboy import PyBoy
from enum import Enum

from collections import defaultdict

# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'growing', 2: 'big', 3: 'shrinking', 4 : 'invincibility blinking'})


class Actions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    B = 5
    A = 6
    # LONG_JUMP = 7

matrix_shape = (4, 16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint16)

class MarioEnv(gym.Env):
    def __init__(self, pyboy, debug=False):
        super().__init__()
        self.rom = "rom.gb"
        self.pyboy = pyboy(self.rom, window='null')
        
        self._x_position_last = 0
        
        self.debug = debug
        self.float16 = False
        
        if not self.debug:
            self.pyboy.set_emulation_speed(1)

        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = game_area_observation_space

        self.pyboy.game_wrapper.start_game()  
    
    @property
    def _x_position(self):
        return self.pyboy.memory[0xC202] * 0x100
    
    @property
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.pyboy.memory[0xFF99]]

        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == Actions.NOOP.value:
            pass
        elif action == Actions.LEFT.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT.value:
            self.pyboy.button_press("right")
        elif action == Actions.UP.value:
            self.pyboy.button("up")
        elif action == Actions.DOWN.value:
            self.pyboy.button("down")
        elif action == Actions.B.value:
            self.pyboy.button_press("b")
            self.pyboy.button_release("b")
        elif action == Actions.A.value:
            self.pyboy.button("a")
        # elif action == Actions.LONG_JUMP.value:
        #     self.pyboy.button_press("b")
        #     self.pyboy.button("a")
        #     self.pyboy.button_press("right")
            
            

        self.pyboy.tick()

        done = self.pyboy.game_wrapper.game_over()
        
        self._x_position_last = self._x_position
        reward = self._x_position - self._x_position_last
        
        
        
        observation=self._get_obs()
        info = self.pyboy.game_wrapper
        truncated = False

        return observation, reward, done, truncated, info
    
    
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.pyboy.game_wrapper.reset_game()
        
        self._x_position_last = 0
        
        observation=self._get_obs()
        info = {}
        return observation, info

    def render(self):
        self.pyboy.tick()
                

    def close(self):
        self.pyboy.stop()

    def _get_obs(self):
        self.pyboy.game_wrapper.game_area_mapping(self.pyboy.game_wrapper.mapping_compressed, 0)
        return self.pyboy.game_area()
    
    # def load_state(self):
    #     with open("state_file.state", "rb") as f:
    #         self.pyboy.load_state(f)
        
# env = MarioEnv(PyBoy)

# print(env.observation_space.shape)

# state, info = env.reset()

# for i in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(f"{info}")
#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()

