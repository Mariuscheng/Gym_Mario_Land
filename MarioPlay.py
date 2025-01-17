import torch
from MarioMetricLogger import MarioMetricLogger
from MarioAgent import MarioAgent
from MarioEnv import MarioEnv
from MarioWrapper import SkipFrame

from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation
import numpy as np
from pyboy import PyBoy

# Function to find the latest checkpoint in the checkpoints directory
def find_latest_checkpoint(save_dir):
    all_checkpoints = list(save_dir.glob('**/mario_net_*.chkpt'))
    if all_checkpoints:
        return max(all_checkpoints, key=os.path.getmtime)
    return None


model_name= "Resume_working"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pyboy = PyBoy("rom.gb", window="SDL2")    
#pyboy.set_emulation_speed(0)

#mario.start_game()
# 初始化 Mario 環境
env = MarioEnv(pyboy)
mario = pyboy.game_wrapper

env = SkipFrame(env, skip=4)
env = FrameStackObservation(env, stack_size=4)

base_save_dir = Path("checkpoints")
save_dir = base_save_dir / model_name #datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

mario_agent = MarioAgent(state_dim=(4, 16, 20), action_dim = env.action_space.n, save_dir=save_dir)

latest_checkpoint = find_latest_checkpoint(save_dir)

current_episode = 0

if latest_checkpoint:
    print(f"Found latest checkpoint at {latest_checkpoint}. Resuming from this checkpoint.")
    mario_agent.load(latest_checkpoint)
    logger = MarioMetricLogger(save_dir,resume=True)
    current_episode = mario_agent.curr_episode
else:
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = MarioMetricLogger(save_dir)
    print("No existing checkpoints found. Created a new directory for this training session.")
    

# mario.game_area_mapping(mario.mapping_compressed, 0)
mario.set_world_level(1, 2)
mario.start_game()

assert mario.lives_left == 2
assert mario.time_left == 400
assert mario.coins == 0
assert mario.score == 0


episodes = 40000
print("Starting from episode",current_episode)
while current_episode < episodes:
    
    observation, info = env.reset()
    
    mario.set_lives_left(10)
    
    while True:
    # for i in range(1000):
        assert mario.time_left <= mario.time_left
        last_time = mario.time_left
        
        mario_score = mario.score
        mario_coins = mario.coins
        
        
        action = mario_agent.act(observation)
        #action = env.action_space.sample()  # 隨機選擇動作
        #print("Action: ", action)  # 隨機選擇動作
        # Example Assuming env is already defined and action is selected 
        next_state, reward, terminated, truncated, info = env.step(action)

        mario_agent.cache(observation, next_state, action, reward, terminated)
    
        # Initialize with the current progress
        # ... inside your game loop
        current_progress = max(0, mario.level_progress)
        
        if current_progress > mario.level_progress:
            reward += 1
        
        blank = np.atleast_1d([300])
        tree =  np.atleast_2d([360, 361, 361, 361, 361, 362])    
        floor = np.atleast_1d([352])
        question_block = np.atleast_1d([129])
        tube = np.atleast_2d([[368,369],
                               [370,371]])
        mario_array = np.atleast_2d([[0,1],
                               [16,17]])

        if mario_coins + 1:
            mario_score + 100
            
        if mario.lives_left -1:
            reward += 0
            
        if mario.lives_left == 0:
            reward == 0
            mario.reset_game()
            break
        
        mario_X, mario_y = pyboy.memory[0XC202], pyboy.memory[0XC201]
        
        power_status_time = pyboy.memory[0XFFA6]
        if power_status_time == 0x90:
            mario_score += 0
        
        power_status = pyboy.memory[0XFF99]
        if power_status == 0X02:
            mario_score += 1000
        
        Goomba = np.atleast_1d([144])
        if Goomba == np.zeros(1).all(where=True):
        # if (Goomba == 0).all():
            mario_score += 100
            
        #if (turle == 0).all():
        turle = np.atleast_2d([[150],
                            [151]])
        if np.all(turle == np.zeros((2, 1)), where=True):
            mario_score += 100
            
        flying_1 = np.atleast_2d([[160,161],
                                [176,177]])
        if np.all(flying_1 == np.zeros((2, 2)), where=True):
            mario_score += 400
        
        flying_2 = np.atleast_2d([[192,193],
                                [208,209]])
        if np.all(flying_2 == np.zeros((2, 2)), where=True):
            mario_score += 800
            
                
        reward = mario_score
              
        # Update state
        observation = next_state
        
        terminated = {}
        #truncated = mario.level_progress >= 2601 or mario.time_left == 0
        truncated = mario.level_progress >= 2601

        if truncated == True:
            reward + 1000
            print("level complete")
            pyboy.stop()
            break
        
        # Learn
        q, loss = mario_agent.learn()
        
        # Logging
        logger.log_step(reward, loss, q)
        
    logger.log_episode()

    if (current_episode % 20 == 0) or (current_episode == episodes - 1):
        logger.record(episode=current_episode, epsilon=mario_agent.exploration_rate, step=mario_agent.curr_step)
    current_episode+=1
    mario_agent.curr_episode = current_episode

