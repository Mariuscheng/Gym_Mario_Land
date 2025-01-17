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
        # current_progress = max(0, mario.level_progress)
        
        # if current_progress > mario.level_progress:
        #     reward += 1
        
        blank = np.argwhere([300])
        tree =  np.argwhere([next_state ==  360, next_state == 361, next_state == 362])    
        floor = np.argwhere([[next_state == 352],[next_state == 353]])
        
        question_block = np.argwhere([next_state ==  129])
        tube = np.argwhere([[next_state ==  368, next_state ==  369],
                            [next_state ==  370, next_state ==  371]])
        mario_array = np.argwhere([[next_state == 8, next_state ==  9],
                                    [next_state == 24, next_state == 25]])
        if len(mario_array) == 0:
            mario.lives_left - 1
            
        Tatol_coins = mario_coins + 1
        mario_score = Tatol_coins*100

        if mario.lives_left == 0:
            mario.reset_game()
            break
        
        Goomba = np.argwhere(next_state == 144)
        if len(Goomba) == 0:
            mario_score + 100
                
        turle = np.argwhere([[next_state == 150], [next_state == 151]])
        if len(turle) == 0:
            mario_score + 100
            
        flying_1 = np.argwhere([[next_state == 160, next_state == 161],
                                [next_state == 176, next_state == 177]])
        if len(flying_1) == 0:
            mario_score + 400
        
        flying_2 =  np.argwhere([[next_state == 192, next_state == 193],
                                [next_state == 208, next_state == 209]])
        if len(flying_2) == 0:
            mario_score + 800
            
                
        mario_socere = reward
              
        # Update state
        observation = next_state
        
        terminated = {}
        #truncated = mario.level_progress >= 2601 or mario.time_left == 0
        truncated = mario.level_progress >= 2601

        if truncated == True:
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
