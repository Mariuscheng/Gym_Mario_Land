import gymnasium as gym
import numpy as np
from collections import deque


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, truncated, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # Update observation space to handle stacked frames
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_stack, *self.env.observation_space.shape),
            dtype=np.uint16
        )

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation(), info

    def _get_observation(self):
        return np.stack(self.frames, axis=0)