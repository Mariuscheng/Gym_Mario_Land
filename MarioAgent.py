import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from MarioNet import DuelingMarioNet

import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress inductor/triton errors

class MarioAgent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.best_reward = float('-inf')

        # Ensure CUDA is used if available
        # Remove torch.compile() and keep basic CUDA setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        self.net = DuelingMarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 0.5
        self.exploration_rate_decay = 0.9999
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.curr_episode = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        # Initialize replay buffer - keep on CPU for memory efficiency
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(100000, device=torch.device("cpu"))
        )
        self.batch_size = 64

        self.gamma = 0.99

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 5e4  # min. experiences before training
        self.learn_every = 4  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Choose an epsilon-greedy action and update exploration rate.
        """
        # Exploration
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        # Exploitation
        else:
            state = state[0] if isinstance(state, tuple) else state
            with torch.no_grad():
                if not isinstance(state, torch.Tensor):
                    state = torch.from_numpy(np.asarray(state, dtype=np.float32))
                state = state.to(self.device).unsqueeze(0)
                
                # Simple forward pass without autocast
                action_values = self.net(state, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()
        
        # Update exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        
        # Process on CPU to save GPU memory
        state = np.asarray(first_if_tuple(state), dtype=np.float32)
        next_state = np.asarray(first_if_tuple(next_state), dtype=np.float32)

        # Convert to tensors on CPU first
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([1 if done else 0], dtype=torch.int)

        # Add to memory (stays on CPU)
        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done
        }, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory and move to GPU
        """
        batch = self.memory.sample(self.batch_size)
        # Move batch to GPU if available
        batch = batch.to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
            n_step = 3  # Use 3-step returns
            gamma_n = self.gamma ** n_step
            
            next_state_Q = self.net(next_state, model="online")
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.net(next_state, model="target")[
                np.arange(0, self.batch_size), best_action
            ]
            return (reward + (1 - done.float()) * gamma_n * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(td_estimate, td_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
            # Soft update of target network
            tau = 0.01  # Soft update parameter
            for target_param, online_param in zip(self.net.target.parameters(), self.net.online.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def save(self, current_reward=None):
        # Regular checkpoint save
        checkpoint_path = (
            self.save_dir / f"Mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        
        save_dict = {
            'model_online': self.net.online.state_dict(),
            'model_target': self.net.target.state_dict(),
            'exploration_rate': self.exploration_rate,
            'curr_step': self.curr_step,
            'curr_episode': self.curr_episode,
            'optimizer': self.optimizer.state_dict(),
            'reward': current_reward
        }
        
        # Save regular checkpoint
        torch.save(save_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # If this is the best model so far, save it separately
        if current_reward is not None and current_reward > self.best_reward:
            self.best_reward = current_reward
            best_path = self.save_dir / "mario_net_best.chkpt"
            torch.save(save_dict, best_path)
            print(f"New best model saved with reward: {current_reward}")

    def load(self, load_path, load_best=False):
        if load_best:
            load_path = self.save_dir / "mario_net_best.chkpt"
        
        if not load_path.exists():
            print(f"No checkpoint found at {load_path}")
            return

        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # Load network states
            self.net.online.load_state_dict(checkpoint['model_online'])
            self.net.target.load_state_dict(checkpoint['model_target'])
            
            # Load training state
            self.exploration_rate = checkpoint['exploration_rate']
            self.curr_step = checkpoint['curr_step']
            self.curr_episode = checkpoint['curr_episode']
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load best reward if available
            if 'reward' in checkpoint:
                self.best_reward = checkpoint['reward']
            
            print(f"Loaded checkpoint from {load_path}")
            print(f"Current episode: {self.curr_episode}")
            print(f"Current step: {self.curr_step}")
            print(f"Exploration rate: {self.exploration_rate}")
            if 'reward' in checkpoint:
                print(f"Model reward: {checkpoint['reward']}")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")

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
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)