import gymnasium as gym


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return observation, reward, done, truncated, info