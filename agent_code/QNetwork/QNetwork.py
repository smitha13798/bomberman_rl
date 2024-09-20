import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from collections import deque
import random
import numpy as np
import logging

class QN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QN, self).__init__()
        self.fc1 = nn.Linear(350, 256)  # Input layer
        self.fc2 = nn.Linear(256, 256)         # Hidden layer
        self.fc3 = nn.Linear(256, output_dim) # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action


class QNAgent:
    def __init__(self, state_dim, action_dim, device=None, replay_buffer_capacity=10000, batch_size=64,
                 gamma=0.99, lr=0.001, exploration_max=1.0, exploration_min=0.01, exploration_decay=0.995):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.lr = lr

        self.rewards_history = []

        # Initialize the Q-network (deep Q-network approximation)
        self.q_network = QN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer to store experience
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)  # Random action
            logging.info(f"Random action selected: {action}")
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
            logging.info(f"Greedy action selected: {action}")
        return action
 # Greedy action based on Q-values

    def update(self):
        # Ensure we have enough samples in the buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Get current Q-values
        q_values = self.q_network(state_batch).gather(1, action_batch)

        # Calculate target Q-values using the Bellman equation
        with torch.no_grad():
            next_q_values = self.q_network(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute the loss (Mean Squared Error)
        loss = F.mse_loss(q_values, target_q_values)

        # Perform optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the exploration rate
        self.exploration_rate = exploration_max  # Start at a high exploration rate
        self.exploration_decay = 0.999

    def store_transition(self, state, action, reward, next_state, done):
        # Store the experience in the replay buffer
        self.replay_buffer.store(state, action, reward, next_state, done)

# Replay buffer class to store and sample transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])
    
    # Convert each component to a numpy array
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
    
    # Handle next_state to ensure it's a homogeneous array
        next_state = np.array([np.zeros_like(state[0]) if s is None else s for s in next_state])
        done = np.array(done)

        return state, action, reward, next_state, done


    def __len__(self):
        return len(self.buffer)