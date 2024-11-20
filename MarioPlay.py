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
#pyboy = PyBoy("pinball.gbc",game_wrapper=False)

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


assert mario.lives_left == 2

# Initialize tracking variables
episode_reward = 0
previous_lives = mario.lives_left
previous_level_progress = 0  # Initialize previous_level_progress
total_level_progress = 0  # Track accumulated progress

episodes = 40000
print("Starting from episode",current_episode)
while current_episode < episodes:

    state = env.reset()
    done = False
    episode_reward = 0

    while not done:

        # Run agent on the state
        action = mario_agent.act(state)

        # Agent performs action
        next_state, reward, done, truncated, info = env.step(action)

        # Remember
        mario_agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario_agent.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Track progress and calculate rewards
        if mario.level_progress > previous_level_progress:
            # Add the progress difference to total
            progress_diff = mario.level_progress - previous_level_progress
            total_level_progress += progress_diff
            
            # Base progress reward
            progress_reward = progress_diff * 2.0  # Base multiplier for progress
            
            # Bonus for passing 251 progress
            if mario.level_progress > 251:
                progress_reward += 10.0  # Bigger bonus for passing 251
            
            # Bonus for accumulated progress
            if total_level_progress > 500:
                progress_reward += 15.0  # Bonus for sustained progress
            elif total_level_progress > 250:
                progress_reward += 7.5  # Smaller bonus for medium progress
        else:
            progress_reward = 0

        # Update previous progress
        previous_level_progress = mario.level_progress

        # Update state
        state = next_state

        # Get initial world/level at start
        current_world, current_level = mario.world

        # Check world/level progression
        new_world, new_level = mario.world
        if new_world > current_world or new_level > current_level:
            print(f"Level completed! Advanced to World {new_world}-{new_level}")
            current_world, current_level = new_world, new_level

        # Check for game completion (4-3 -> 4-4)
        if current_world == 4 and current_level == 4:
            print("Game completed! Reached World 4-4")
            break

        episode_reward += reward

    # In the training loop
    # Check for life loss and apply penalty
    if mario.lives_left < previous_lives:
        lives_reward = -100  # Penalty for losing a life
    #else:
        #lives_reward = mario.lives_left * 10  # Bonus for maintaining lives

    previous_lives = mario.lives_left  # Update lives count for next check
    score_reward = mario.score * 0.1

    # Add final rewards to logger
    logger.log_step(
        reward=progress_reward + lives_reward + score_reward,
        loss=loss,
        q=q
    )

    # Log episode
    logger.log_episode()

    # Print episode summary
    world, level = mario.world
    print(f"\nEpisode {current_episode} - Game Over!")
    print(f"World {world}-{level}")
    print(f"Progress: {mario.level_progress}")
    print(f"Total Progress Accumulated: {total_level_progress}")
    print(f"Lives Remaining: {mario.lives_left}")
    print(f"Score: {mario.score}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

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
    current_episode += 1
    mario_agent.curr_episode = current_episode