# Thesis Project: Optimizing Wireless Network Resources with Deep Reinforcement Learning

### Work in Progress... will add more throughout the development of the Thesis

## Overview
This project focuses on optimizing wireless network resources using Deep Reinforcement Learning (DRL).
The primary objective is to leverage the Soft Actor-Critic (SAC) algorithm to improve system efficiency in terms of power consumption, user satisfaction, 
and overall network performance of cell-free massive MIMO networks.

## Goals
- Implement a custom wireless network simulation environment with Gym, then later on without needing Gymn.
- Train a SAC agent to learn optimal policies for network resource management.
- Explore the impact of fading, interference, and obstacles on network performance.
- Investigate how throwing an agent into a reinforcement learning environment can adaptively allocate resources based on variable network conditions.

## Technologies
- **Python Library: Gymnasium** for custom environment creation and simulation of DRL techniques.
- **SAC Algorithm** for training the agent to optimize resource allocation.
##### Eventually...
- **PyTorch** for deep reinforcement learning model development.

## Structure
1. **Environment**: A custom wireless network environment with continuous action spaces.
2. **Agent**: A Soft Actor-Critic agent that learns to allocate resources optimally.
3. **Reward Function**: Focus on maximizing entropy and network performance by minimizing interference and power consumption.
