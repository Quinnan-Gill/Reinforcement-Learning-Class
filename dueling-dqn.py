from argparse import Namespace
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim
from collections import namedtuple, deque

from connect_four_env import ConnectFourEnv
from rl_agent import RLModel

class QNetwork(nn.Module):

    def __init__(
        self, state_size, action_size, seed,
        adv_type = 'avg', fc1_units=128, fc2_units=64,fc3_units = 256
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc_adv = nn.Linear(fc2_units, fc3_units)
        self.fc_value = nn.Linear(fc2_units, fc3_units)
        self.adv = nn.Linear(fc3_units, action_size)
        self.value = nn.Linear(fc3_units, 1)
        self.adv_type = adv_type

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x_adv = F.relu(self.fc_adv(x))
        x_adv = F.relu(self.adv(x_adv))
        x_value = F.relu(self.fc_value(x))
        x_value = F.relu(self.adv(x_value))
        if self.adv_type == 'avg':
          advAverage = torch.mean(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advAverage
        else:
          advMax,_ = torch.max(x_adv, dim=1, keepdim=True)
          q =  x_value + x_adv - advMax
        return q
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

