import torch
from MarioMetricLogger import MarioMetricLogger
from MarioAgent import MarioAgent
from MarioEnv import MarioEnv
from MarioWrapper import SkipFrame

from pathlib import Path
from collections import deque
import random, datetime, os
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation, StickyAction
import numpy as np
from pyboy import PyBoy
import sys

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

env = SkipFrame(env, skip=4)
env = FrameStackObservation(env, stack_size=4)
#env = StickyAction(env, repeat_action_probability=0.8)

mario = pyboy.game_wrapper

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
    
    observation, info = env.reset(seed=42)
    mario.set_lives_left(10)
    
    previous_enemy_positions = set()
    
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
        current_progress = mario.level_progress
        current_progress += 1
        
        #game data set and reward
        
        elements_to_zero = [300, 310, 305, 306, 307, 350, 336, 338]
        next_state[np.isin(next_state, elements_to_zero)] = 0
        
        block = np.argwhere(next_state ==  130)
        black_block = np.argwhere(next_state ==  355)
        tree =  np.argwhere(np.isin(next_state, [360, 361, 362]))    
        floor = np.argwhere(np.isin(next_state, [352, 353, 232]))
        question_block = np.argwhere(np.isin(next_state, [129]))
        tree =  np.argwhere(np.isin(next_state, [368, 369, 370, 371]))
        
       
        mario_array = np.argwhere(np.isin(next_state, [8, 9, 24, 25]))
        if not mario_array.size:
        # Potentially handle when Mario is not found, if needed
            mario_score = mario_score + 0
            
        Tatol_coins = (mario_coins + 1)*100
        mario_score = Tatol_coins
        

        if mario.lives_left == 0:
            mario.reset_game()
            break
        
        
        # Identify current enemy positions in the next_state
        Goomba = set(tuple(pos) for pos in np.argwhere(next_state == 144))
        Turtle = set(tuple(pos) for pos in np.argwhere(np.isin(next_state, [150, 151])))
        Flying_1 = set(tuple(pos) for pos in np.argwhere(np.isin(next_state, [160, 161, 176, 177])))
        Flying_2 = set(tuple(pos) for pos in np.argwhere(np.isin(next_state, [192, 193, 208, 209])))

        # Combine all current enemy positions
        current_enemy_positions = Goomba | Turtle | Flying_1 | Flying_2

        # Calculate enemies defeated by checking disappearance from previous state
        enemies_defeated = (previous_enemy_positions - current_enemy_positions)
        
        # Award points for each enemy defeated
        mario_score += 100 * len(enemies_defeated)

        # Update previous enemy positions for the next iteration
        previous_enemy_positions = current_enemy_positions.copy()
        
        # Goomba = np.argwhere(next_state == 144)
        # Goomba = np.argwhere(next_state == 144)
        # if bool(len(Goomba) == 0) == True:
        #     mario_score = mario_score + 100
        # else:
        #     mario_score = mario_score + 0
                
        # turle = np.argwhere(np.isin(next_state, [150, 151]))
        # if np.all(len(turle) == 0):
        #     mario_score = mario_score + 100
        # else:
        #     mario_score = mario_score + 0
            
        # flying_1 = np.argwhere([[next_state == 160, next_state == 161],
        #                         [next_state == 176, next_state == 177]])
        # if np.all(len(flying_1) == 0):
        #     mario_score = mario_score + 400
        # else:
        #     mario_score = mario_score + 0
        
        # flying_2 =  np.argwhere([[next_state == 192, next_state == 193],
        #                         [next_state == 208, next_state == 209]])
        # if np.all(len(flying_2) == 0):
        #     mario_score = mario_score + 800
        # else:
        #     mario_score = mario_score + 0
            
        if pyboy.memory[0xFFA6] == 144:
            mario_score = mario_score + 0
            
        if pyboy.memory[0xFF99] == 1:
            mario_score = mario_score + 1000
        else :
            mario_score = mario_score + 0
            
        bool(pyboy.memory[0xC20A] == 1)
                
        mario_socere = reward
              
        # Update state
        observation = next_state
        
        terminated = {}
        #truncated = mario.level_progress >= 2601 or mario.time_left == 0
        truncated = mario.level_progress >= 2601
        
        # Check if level is complete
        if truncated == True:
            print("level complete")
            pyboy.stop()
            sys.exit(0)
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
