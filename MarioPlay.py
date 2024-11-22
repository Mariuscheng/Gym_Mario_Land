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

class MarioPlay:
    def __init__(self, rom_path="rom.gb", episodes=400):
        self.episodes = episodes
        self.pyboy = PyBoy(rom_path, window="SDL2")
        self.env = MarioEnv(self.pyboy, debug=True)  # Enable debug mode
        self.env = SkipFrame(self.env, skip=4)
        self.env = FrameStackObservation(self.env, stack_size=4)
        self.mario = self.pyboy.game_wrapper
        
        # Setup CUDA
        self.use_cuda = torch.cuda.is_available()
        print(f"Using CUDA: {self.use_cuda}")
        
        # Setup directories and agent
        self.base_save_dir = Path("checkpoints")
        self.save_dir = self.base_save_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.matrix_shape = (4, 16, 20)
        # Calculate total number of possible actions from MultiDiscrete space
        total_actions = np.prod(self.env.action_space.nvec)
        self.mario_agent = MarioAgent(
            state_dim=self.matrix_shape, 
            action_dim=total_actions,
            save_dir=self.save_dir
        )
        
        # Initialize logger
        self.setup_logger()
        self.save_parameters()
        self.current_episode = self.load_checkpoint()

    def get_mario_position(self):
        level_block = self.pyboy.memory[0xC0AB]
        mario_x = self.pyboy.memory[0xC202]
        return level_block * 16 + mario_x

    def setup_logger(self):
        if self.save_dir.exists():
            self.logger = MarioMetricLogger(self.save_dir, resume=True)
            print(f"Loading existing checkpoint directory: {self.save_dir}")
        else:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.logger = MarioMetricLogger(self.save_dir)
            print(f"Created new checkpoint directory: {self.save_dir}")

    def save_parameters(self):
        with open(self.save_dir / 'parameters.txt', 'w') as f:
            f.write(f"Episodes: {self.episodes}\n")
            f.write(f"Exploration Rate: {self.mario_agent.exploration_rate}\n")
            f.write(f"Exploration Rate Decay: {self.mario_agent.exploration_rate_decay}\n")
            f.write(f"Exploration Rate Min: {self.mario_agent.exploration_rate_min}\n")
            f.write(f"Learning Rate: {self.mario_agent.optimizer.param_groups[0]['lr']}\n")
            f.write(f"Gamma: {self.mario_agent.gamma}\n")
            f.write(f"Memory Size: 100000\n")
            f.write(f"Batch Size: {self.mario_agent.batch_size}\n")
            f.write(f"Learn Every: {self.mario_agent.learn_every}\n")
            f.write(f"Sync Every: {self.mario_agent.sync_every}\n")
            f.write(f"Burn-in: {self.mario_agent.burnin}\n")
            f.write(f"Info: {self.mario}\n")

    def find_latest_checkpoint(self):
        all_checkpoints = list(self.save_dir.glob('**/mario_net_*.chkpt'))
        all_checkpoints = [cp for cp in all_checkpoints if 'best' not in str(cp)]
        if all_checkpoints:
            return max(all_checkpoints, key=os.path.getmtime)
        return None

    def find_best_checkpoint(self):
        best_model = self.save_dir / "mario_net_best.chkpt"
        return best_model if best_model.exists() else None

    def load_checkpoint(self):
        best_checkpoint = self.find_best_checkpoint()
        latest_checkpoint = self.find_latest_checkpoint()
        
        if best_checkpoint:
            print(f"Loading best performing model: {best_checkpoint}")
            self.mario_agent.load(best_checkpoint)
            current_episode = self.mario_agent.curr_episode
        elif latest_checkpoint:
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            self.mario_agent.load(latest_checkpoint)
            current_episode = self.mario_agent.curr_episode
        else:
            print("No existing checkpoints found. Starting fresh training session.")
            current_episode = 0
            
        self.mario.reset_game()
        self.env.reset()
        return current_episode

    def get_rewards(self, prev_state, current_state):
        # Extract relevant information from states
        prev_world, prev_level = prev_state.get('world', (0, 0))
        prev_time = prev_state.get('time_left', 0)
        prev_lives = prev_state.get('lives_left', 0)
        prev_x_pos = prev_state.get('x_pos', 0)
        prev_progress = prev_state.get('level_progress', 0)
        
        curr_world, curr_level = current_state.get('world', (0, 0))
        curr_time = current_state.get('time_left', 0)
        curr_lives = current_state.get('lives_left', 0)
        curr_x_pos = current_state.get('x_pos', 0)
        curr_progress = current_state.get('level_progress', 0)
        
        # Calculate rewards
        reward = 0
        
        # Progress reward based on both local and global position
        x_movement = curr_x_pos - prev_x_pos
        progress_movement = curr_progress - prev_progress
        
        # Reward forward movement using both metrics
        if progress_movement > 0:
            reward += progress_movement * 0.5  # Reward for global progress
        if x_movement > 0:
            reward += x_movement * 0.3  # Additional reward for local movement
        
        # Time penalty - smaller penalty
        time_penalty = (prev_time - curr_time) * 0.05
        reward -= time_penalty
        
        # Life loss penalty
        if curr_lives < prev_lives:
            reward -= 25
        
        # Level completion reward
        if curr_world > prev_world or curr_level > prev_level:
            reward += 100
        
        # Add reward for killing enemies
        killable_by_jumping = self.pyboy.memory[0xD109]
        if killable_by_jumping:
            reward += 15
        
        return reward

    def training(self):
        print(f"Starting training from episode {self.current_episode}")
        
        best_reward = float('-inf')
        
        while self.current_episode < self.episodes:
            state = self.env.reset()
            done = False
            episode_reward = 0
            last_x_pos = self.get_mario_position()
            stuck_counter = 0
            
            # Track loss values for this episode
            episode_losses = []

            while not done:
                # Get action from agent
                action = self.mario_agent.act(state)
                
                # Store previous state info
                prev_state = {
                    'world': self.mario.world,
                    'time_left': self.mario.time_left,
                    'lives_left': self.mario.lives_left,
                    'x_pos': last_x_pos,
                    'level_progress': self.mario.level_progress
                }
                
                # Perform action
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # Get current position and check if stuck
                current_x_pos = self.get_mario_position()
                if abs(current_x_pos - last_x_pos) < 2:
                    stuck_counter += 1
                    if stuck_counter > 1000:
                        print("Mario appears to be stuck, resetting game...")
                        self.mario.reset_game()
                        done = True
                else:
                    stuck_counter = 0
                
                # Check if Mario lost all lives
                if self.mario.lives_left <= 0:
                    print("Game Over - Mario lost all lives. Resetting game...")
                    self.mario.reset_game()
                    done = True
                
                # Calculate rewards
                current_state = {
                    'world': self.mario.world,
                    'time_left': self.mario.time_left,
                    'lives_left': self.mario.lives_left,
                    'x_pos': current_x_pos,
                    'level_progress': self.mario.level_progress
                }
                reward = self.get_rewards(prev_state, current_state)
                
                # Learn from experience
                self.mario_agent.cache(state, next_state, action, reward, done)
                learn_result = self.mario_agent.learn()
                
                # Handle the learning results
                if learn_result is not None:
                    q_value, loss = learn_result
                    if loss is not None:
                        episode_losses.append(loss)
                else:
                    q_value, loss = None, None
                
                # Log step with actual reward and q_value/loss
                self.logger.log_step(reward, loss if loss is not None else 0.0, q_value if q_value is not None else 0.0)
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                last_x_pos = current_x_pos

            # Calculate average loss for this episode
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

            # Log episode metrics
            self.logger.log_episode()
            
            # Print episode summary before incrementing episode number
            print(f"Episode {self.current_episode} - Reward: {episode_reward:.2f}, Loss: {avg_loss:.3f}, World: {self.mario.world}")
            
            # Record metrics and increment episode
            self.logger.record(
                episode=self.current_episode,
                epsilon=self.mario_agent.exploration_rate,
                step=self.mario_agent.curr_step
            )
            
            # Save model if it's the best so far
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.mario_agent.save(current_reward=episode_reward)
            
            self.current_episode += 1

        # Save final model
        print("\nTraining complete! Saving final model...")
        self.mario_agent.save(current_reward=episode_reward)
        print(f"Final model saved with reward: {episode_reward:.2f}")
        
        self.pyboy.stop()
        
def main():
    mario_play = MarioPlay()
    mario_play.training()

if __name__ == "__main__":
    main()
