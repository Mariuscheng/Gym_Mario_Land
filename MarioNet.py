import torch
import torch.nn as nn
import torch.optim as optim

# from gymnasium.wrappers import FlattenObservation

# from MarioEnv import MarioEnv

# from pyboy import PyBoy

# pyboy = PyBoy("rom.gb", window="null")
# env = MarioEnv(pyboy)
# env = FlattenObservation(env)

class DuelingMarioNet(nn.Module):
    """Dueling DQN architecture
    Splits into value and advantage streams for better policy evaluation
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 16:
            raise ValueError(f"Expecting input height: 16, got: {h}")
        if w != 20:
            raise ValueError(f"Expecting input width: 20, got: {w}")

        self.online = self.__build_dueling_network(c, output_dim)
        self.target = self.__build_dueling_network(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False
            
        self.output_dim = output_dim

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def evaluate(self, eval_loader, device):
        """
        Evaluate model performance on a validation set
        
        Args:
            eval_loader: DataLoader containing validation data
            device: Device to run evaluation on
        
        Returns:
            dict containing evaluation metrics
        """
        self.eval()  # Set to evaluation mode
        total_value_loss = 0
        total_advantage_loss = 0
        total_samples = 0
        correct_actions = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                states, actions, rewards, next_states, dones = batch
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                
                # Get Q-values from online network
                q_values = self.online(states)
                predicted_actions = q_values.argmax(dim=1)
                
                # Calculate accuracy
                correct_actions += (predicted_actions == actions).sum().item()
                total_samples += actions.size(0)
                
                # Get value and advantage estimates
                features = self.online.feature_net(states)
                values = self.online.value_net(features)
                advantages = self.online.advantage_net(features)
                
                # Calculate TD error as a proxy for value estimation quality
                next_q_values = self.target(next_states)
                next_values = next_q_values.max(1)[0]
                expected_values = rewards + (1 - dones) * 0.99 * next_values
                value_loss = ((values - expected_values) ** 2).mean()
                
                total_value_loss += value_loss.item()
                
        # Calculate metrics
        action_accuracy = correct_actions / total_samples
        avg_value_loss = total_value_loss / len(eval_loader)
        
        return {
            'action_accuracy': action_accuracy,
            'value_loss': avg_value_loss,
        }
    
    def get_action_distribution(self, state, device):
        """
        Get probability distribution over actions for a given state
        
        Args:
            state: Input state tensor
            device: Device to run inference on
            
        Returns:
            Action probabilities
        """
        self.eval()
        with torch.no_grad():
            state = state.to(device)
            q_values = self.online(state)
            # Convert Q-values to probabilities using softmax
            action_probs = torch.softmax(q_values, dim=1)
            return action_probs
    
    def get_value_confidence(self, state, device):
        """
        Calculate confidence in value estimation for a given state
        
        Args:
            state: Input state tensor
            device: Device to run inference on
            
        Returns:
            Value estimate and confidence score
        """
        self.eval()
        with torch.no_grad():
            state = state.to(device)
            features = self.online.feature_net(state)
            value = self.online.value_net(features)
            advantage = self.online.advantage_net(features)
            
            # Calculate confidence based on advantage spread
            advantage_spread = advantage.max(dim=1)[0] - advantage.min(dim=1)[0]
            confidence = 1.0 / (1.0 + advantage_spread)  # Higher spread = lower confidence
            
            return value, confidence

    def __build_dueling_network(self, c, output_dim):
        # Calculate the output size of conv layers
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # Calculate output dimensions after conv layers
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(16)))  # Height after 3 conv layers
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(20)))  # Width after 3 conv layers
        linear_input_size = 64 * conv_h * conv_w  # 64 channels from last conv layer

        # Shared feature layers
        feature_net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten()
        )

        # Value stream
        value_net = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)  # Single value for state
        )

        # Advantage stream
        advantage_net = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)  # Advantage for each action
        )

        class DuelingNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_net = feature_net
                self.value_net = value_net
                self.advantage_net = advantage_net

            def forward(self, x):
                features = self.feature_net(x)
                value = self.value_net(features)
                advantage = self.advantage_net(features)
                
                # Combine value and advantage using the dueling formula
                # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
                return value + (advantage - advantage.mean(dim=1, keepdim=True))

        return DuelingNetwork()


# print(env.observation_space.shape[0])
# print(env.action_space.n)

# n_observations = env.observation_space.shape[0] #320
# n_actions = env.action_space.n #8

# # Initialize the model, optimizer, and loss function
# model = DuelingMarioNet(input_dim=n_observations, output_dim=n_actions)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = nn.MSELoss()

# print(model)
# MarioNet(
#   (fc1): Linear(in_features=320, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=128, bias=True)
#   (fc3): Linear(in_features=128, out_features=8, bias=True)
# )

# print(optimizer)
# Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     lr: 0.0001
#     maximize: False
#     weight_decay: 0
# )

# print(loss_fn)
# MSELoss()


# observations , info = env.reset()
# print(observations)

# # Example usage:
# observations = torch.tensor(observations, dtype=torch.float32)
# q_values = model(observations)

# print(q_values)
# [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 13  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0 14 14  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 14 14  0
#   1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10 10 10 10 10 10 10 10
#  10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
#  10 10 10 10 10 10 10 10]
# tensor([-0.2241,  0.5682, -0.0036, -0.4232, -0.1634, -0.2661,  0.2863, -0.4242],
#        grad_fn=<ViewBackward0>)
