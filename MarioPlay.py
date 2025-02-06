import torch
from MarioMetricLogger import MarioMetricLogger
from MarioAgent import MarioAgent
from MarioEnv import MarioEnv

from pathlib import Path
from collections import deque
import os
from gymnasium import spaces
from gymnasium.wrappers import FrameStackObservation
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

#env = SkipFrame(env, skip=4)
env = FrameStackObservation(env, stack_size=4)
#env = StickyAction(env, repeat_action_probability=0.8)


mario = pyboy.game_wrapper
#mario.set_world_level(3, 2)
mario.start_game()

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
    

#mario.game_area_mapping(mario.mapping_compressed, 0)

# assert mario.lives_left == 2
assert mario.time_left == 400
assert mario.coins == 0
assert mario.score == 0

# while pyboy.tick():
#     print(mario)
#     print(pyboy.get_sprite_by_tile_identifier([16,17]))
#     pass
    
# pyboy.stop()

episodes = 40000
print("Starting from episode",current_episode)
while current_episode < episodes:
    
    observation, info = env.reset(seed=42)
    
    mushroom = pyboy.get_sprite_by_tile_identifier([131])
        
    mario_pos = pyboy.get_sprite_by_tile_identifier([0, 1, 16, 17], on_screen=True)
    Big_mario = pyboy.get_sprite_by_tile_identifier([33, 32, 49, 48])
    flying_1 = pyboy.get_sprite_by_tile_identifier([160, 161, 176, 177])
    tube = pyboy.get_sprite_by_tile_identifier([368, 369, 370, 371])    
    floor = pyboy.get_sprite_by_tile_identifier([352, 353])   
    Goomba = pyboy.get_sprite_by_tile_identifier([144])
    turle = pyboy.get_sprite_by_tile_identifier([150, 151])         
    blank = pyboy.get_sprite_by_tile_identifier([300])
    
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
        current_progress = mario.level_progress
        current_progress += 1
        
        Tatol_coins = (mario_coins + 1)*100
        mario_score = Tatol_coins
            
        
        #flying_2 = jax.numpy.argwhere(jnp.isin(observation_tensor , jnp.array([192, 193, 208, 209])))
        # if flying_2.size != 0:
        #     if flying_2.size == 0:
        #         print("flying_2 : ", flying_2)
        #         mario_score = mario_score + 800
        
        # Powerup Status Timer    
        if 2 <= pyboy.memory[0xFFA6] < 144:
            mario_score = mario_score + 0
        
        # Y position
        mario_y_pos = pyboy.memory[0XC202]
        
        # jump routine
        jump_routine = pyboy.memory[0XC207]
        
        # something move
        move_thing = pyboy.memory[0XD103]
        if move_thing == 0:
            mario_score = mario_score + 100 
        
        # Powerup Status  
        Powerup_Status = pyboy.memory[0xFF99]
        if Powerup_Status == 0:
            mario_score = mario_score + 0

        if mario.lives_left == 0:
            mario.reset_game()
            # print(jnp.isin(observation, jnp.array([144])))
            break
              
        reward = mario_score
        # print(reward)
              
        # Update state
        observation_tensor = next_state
        
        terminated = mario.level_progress >= 2601
        #truncated = mario.level_progress >= 2601 or mario.time_left == 0
        truncated = {}
        
        
        # Check if level is complete
        if terminated == True:
            mario_score = max(0, mario_score)
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
