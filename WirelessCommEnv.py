import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# WirelessNetworkCommsEnv
class WirelessNetworkEnv:
    def __init__(self, num_aps=5, num_users=10):
        self.num_aps = num_aps
        self.num_users = num_users
        # State dimension is AP and user count
        self.state_dim = num_aps + num_users
        # Actions represent power allocation for APs
        self.action_dim = num_aps
        self.max_power = 10.0  # Max power per AP
        self.interference_factor = 0.1
        self.fading = np.random.uniform(0.5, 1.5, size=(num_users, num_aps))
        self.reset()

    def reset(self):
        # Random initial state
        self.state = np.random.uniform(0, 1, size=self.state_dim)
        return self.state

    def step(self, action):
        power_alloc = np.clip(action, 0, self.max_power)
        interference = self.interference_factor * np.sum(power_alloc)
        user_satisfaction = np.sum(self.fading @ power_alloc) / self.num_users

        # Reward: Maximize throughput while minimizing power consumption and interference
        reward = user_satisfaction - interference - np.sum(power_alloc) * 0.01

        self.fading = np.random.uniform(0.5, 1.5, size=(self.num_users, self.num_aps)) # Fade channel randomly
        self.state = np.random.uniform(0, 1, size=self.state_dim)
        done = False  # False requires that the agent stays continuous in the environment
        return self.state, reward, done

# Next implementation steps

# SAC implementation in PyTorch
class SACAgent:
    def __init__(self, state_dim, action_dim):
        pass
    # Policy Network
    # Q Network
    # SAC Agent

# Training the SAC agent
env = WirelessNetworkEnv()
agent = SACAgent(state_dim = env.state_dim, action_dim = env.action_dim)