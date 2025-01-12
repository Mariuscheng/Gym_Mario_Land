import numpy as np
import torch
from collections import deque
from MarioNet import MarioNet
# from MarioEnv import A_FUNCTION, B_FUNCTION, ARROW_Function
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import cv2
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class MarioAgent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.curr_episode = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        
        self.batch_size = 32

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, observation):
        """
    Given a observation, choose an epsilon-greedy action and update value of step.

    Inputs:
    observation(``LazyFrame``): A single observation of the current observation, dimension is (state_dim)
    Outputs:
    ``action_idx`` (``int``): An integer representing which action Mario will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            observation = observation[0] if isinstance(observation, tuple) else observation
            observation = np.asarray(observation, dtype=np.float32)
            observation = torch.tensor(observation, device=self.device).unsqueeze(0)
            action_values = self.net(observation, model="online")
            action_idx = torch.argmax(action_values,dim=1).item()
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, observation, next_state, action, reward, terminated):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        observation (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        observation = np.asarray(first_if_tuple(observation),dtype=np.float32)
        next_state = np.asarray(first_if_tuple(next_state),dtype=np.float32)

        observation = torch.tensor(observation).to(self.device)
        next_state = torch.tensor(next_state).to(self.device)
        action = torch.tensor([action]).to(self.device)
        reward = torch.tensor([reward]).to(self.device)
        terminated = torch.tensor([bool(terminated)], dtype=torch.bool).to(self.device)

        #self.memory.append((observation, next_state, action, reward, terminated,))
        self.memory.add(TensorDict({"observation": observation, "next_state": next_state, "action": action, "reward": reward, "terminated": terminated}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        observation, next_state, action, reward, terminated = (batch.get(key) for key in ("observation", "next_state", "action", "reward", "terminated"))
        return observation, next_state, action.squeeze(), reward.squeeze(), terminated.squeeze()

    def td_estimate(self, observation, action):
        current_Q = self.net(observation, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - terminated.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        print("saving current episode")
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, curr_step=self.curr_step, curr_episode=self.curr_episode,),
            save_path,
        )
        print(f"mario Net saved to {save_path} at step {self.curr_step}")
        if self.curr_episode % 20 == 0:
            quit()

    def load(self, load_path):
        if load_path.is_file():
            checkpoint = torch.load(load_path,weights_only=True,map_location=torch.device("cuda"))
            self.net.load_state_dict(checkpoint['model'])
            self.exploration_rate = checkpoint['exploration_rate']
            self.curr_step = checkpoint['curr_step']
            self.curr_episode = checkpoint['curr_episode']
            print("loaded current episode",self.curr_episode)
            print(f"Loaded the model from {load_path} with exploration rate = {self.exploration_rate}")
        else:
            print(f"No checkpoint found at {load_path}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        observation, next_state, action, reward, terminated = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(observation, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, terminated)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)