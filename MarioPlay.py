import os
import torch
from pathlib import Path
import datetime

from pyboy import PyBoy
from MarioEnv import MarioEnv
from MarioNet import DuelingMarioNet
from MarioAgent import MarioAgent
from MarioWrapper import SkipFrame
from MarioMetricLogger import MarioMetricLogger
from gymnasium.wrappers import FrameStackObservation
from gymnasium import spaces
import numpy as np
from collections import deque
import random

# Function to find the latest checkpoint in the checkpoints directory
def find_latest_checkpoint(save_dir):
    all_checkpoints = list(save_dir.glob('**/mario_net_*.chkpt'))
    # Exclude best model from regular checkpoints
    all_checkpoints = [cp for cp in all_checkpoints if 'best' not in str(cp)]
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None

def find_best_checkpoint(save_dir):
    best_model = save_dir / "mario_net_best.chkpt"
    return best_model if best_model.exists() else None

pyboy = PyBoy("rom.gb" ,window="SDL2")

env = MarioEnv(pyboy)
env = SkipFrame(env, skip=4)
env = FrameStackObservation(env, stack_size=4)

mario = pyboy.game_wrapper

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

model_name= "Resume_working"

base_save_dir = Path("checkpoints")
save_dir = base_save_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

matrix_shape = (4, 16, 20)
mario_agent = MarioAgent(state_dim=matrix_shape, action_dim=env.action_space.n, save_dir=save_dir)

# Create checkpoint directory
if save_dir.exists():
    logger = MarioMetricLogger(save_dir, resume=True)
    print(f"Loading existing checkpoint directory: {save_dir}")
else:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MarioMetricLogger(save_dir)
    print(f"Created new checkpoint directory: {save_dir}")

# Save initial hyperparameters
with open(save_dir / 'parameters.txt', 'w') as f:
    f.write(f"Episodes: 40000\n")
    f.write(f"Exploration Rate: {mario_agent.exploration_rate}\n")
    f.write(f"Exploration Rate Decay: {mario_agent.exploration_rate_decay}\n")
    f.write(f"Exploration Rate Min: {mario_agent.exploration_rate_min}\n")
    f.write(f"Learning Rate: {mario_agent.optimizer.param_groups[0]['lr']}\n")
    f.write(f"Gamma: {mario_agent.gamma}\n")
    f.write(f"Memory Size: 100000\n")  # Fixed size from LazyMemmapStorage
    f.write(f"Batch Size: {mario_agent.batch_size}\n")
    f.write(f"Learn Every: {mario_agent.learn_every}\n")
    f.write(f"Sync Every: {mario_agent.sync_every}\n")
    f.write(f"Burn-in: {mario_agent.burnin}\n")
    f.write(f"Info: {mario}\n")

latest_checkpoint = find_latest_checkpoint(save_dir)
best_checkpoint = find_best_checkpoint(save_dir)
current_episode = 0

# Load the best model if available
if best_checkpoint:
    print(f"Loading best performing model: {best_checkpoint}")
    mario_agent.load(best_checkpoint)
    current_episode = mario_agent.curr_episode
    # Reset game state after loading checkpoint
    mario = pyboy.game_wrapper
    mario.reset_game()
    state = env.reset()
    print("Game state reset after loading best model")
elif latest_checkpoint:
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    mario_agent.load(latest_checkpoint)
    current_episode = mario_agent.curr_episode
    # Reset game state after loading checkpoint
    mario = pyboy.game_wrapper
    mario.reset_game()
    state = env.reset()
    print("Game state reset after loading checkpoint")
else:
    print("No existing checkpoints found. Starting fresh training session.")

realMax = []
episodes = 400
print("Starting from episode", current_episode)

while current_episode < episodes:
    state = env.reset()
    done = False
    episode_reward = 0

    # Get initial position
    level_block = pyboy.memory[0xC0AB]
    mario_x = pyboy.memory[0xC202]

    while not done:
        # Run agent on the state
        action = mario_agent.act(state)

        # Remember previous state before action
        prev_world, prev_level = mario.world
        prev_time_left = mario.time_left
        prev_lives = mario.lives_left
        prev_x_pos = level_block * 16 + mario_x

        # Agent performs action
        next_state, reward, done, truncated, info = env.step(action)

        # Get current position
        level_block = pyboy.memory[0xC0AB]
        mario_x = pyboy.memory[0xC202]
        current_x_pos = level_block * 16 + mario_x

        # Calculate rewards
        clock = mario.time_left - prev_time_left
        movement = current_x_pos - prev_x_pos
        death = -15 * (mario.lives_left - prev_lives)
        
        current_world, current_level = mario.world
        levelReward = 15 * max((current_world - prev_world), (current_level - prev_level))
        
        # Combine all rewards
        reward = clock + death + movement + levelReward

        # Check respawn timer
        if pyboy.memory[0xFFA6] > 0:
            reward = 0

        # Remember and learn
        mario_agent.cache(state, next_state, action, reward, done)
        q, loss = mario_agent.learn()
        logger.log_step(reward, loss, q)

        # Update level progress max in realMax list
        if len(realMax) == 0:
            realMax.append([current_world, current_level, current_x_pos])
        else:
            found = False
            for entry in realMax:
                if entry[0] == current_world and entry[1] == current_level:
                    entry[2] = max(entry[2], current_x_pos)
                    found = True
                    break
            
            if not found:
                realMax.append([current_world, current_level, current_x_pos])

        # Update state and reward
        state = next_state
        episode_reward += reward

    # End of episode logging
    score_reward = mario.score * 0.1
    logger.log_episode()
    
    # Print episode summary
    world, level = mario.world
    print(f"\nEpisode {current_episode} - Game Over!")
    print(f"World {world}-{level}")
    print(f"Progress: {mario.level_progress}")
    print(f"Lives Remaining: {mario.lives_left}")
    print(f"Score: {mario.score}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    current_episode += 1

    if (current_episode % 20 == 0) or (current_episode == episodes - 1):
        logger.record(
            episode=current_episode,
            epsilon=mario_agent.exploration_rate,
            step=mario_agent.curr_step
        )

    if mario_agent.curr_step % mario_agent.save_every == 0:
        mario_agent.save(current_reward=episode_reward)

    # Reset for next episode
    mario.reset_game()
    mario_agent.curr_episode = current_episode
