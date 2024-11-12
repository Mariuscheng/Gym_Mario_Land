import torch
from MarioMetricLogger import MarioMetricLogger
from MarioAgent import MarioAgent
from MarioEnv import MarioEnv
from MarioNet import MarioNet
from MarioWrappers import SkipFrame
from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces

import numpy as np
from pyboy import PyBoy
#from gymnasium.wrappers import FrameStackObservation

model_name= "Resume_working"

def find_latest_checkpoint(save_dir):
    all_checkpoints = list(save_dir.glob('**/mario_net_*.chkpt'))
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

pyboy = PyBoy("rom.gb",window="null")

env = MarioEnv(PyBoy)
env = SkipFrame(env, skip=4)
#env = FrameStackObservation(env, stack_size=4)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

Mario_Agent = MarioAgent(state_dim=(4, 16, 20), action_dim=env.action_space.n, save_dir=save_dir)

latest_checkpoint = find_latest_checkpoint(save_dir)

current_episode = 0

if latest_checkpoint:
    print(f"Found latest checkpoint at {latest_checkpoint}. Resuming from this checkpoint.")
    Mario_Agent.load(latest_checkpoint)
    logger = MarioMetricLogger(save_dir,resume=True)
    current_episode = Mario_Agent.curr_episode
else:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MarioMetricLogger(save_dir)
    print("No existing checkpoints found. Created a new directory for this training session.")

episodes = 40
print("Starting from episode",current_episode)
while current_episode < episodes:

    state , info = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = Mario_Agent.act(state)
        #print(state)
        # action = torch.tensor(Mario_Agent.act(state))
        

        # Agent performs action
        next_state, reward, done, truncated, info = env.step(action)
       

        # Remember
        Mario_Agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = Mario_Agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode()

    if (current_episode % 20 == 0) or (current_episode == episodes - 1):
        logger.record(episode=current_episode, epsilon=Mario_Agent.exploration_rate, step=Mario_Agent.curr_step)
    
