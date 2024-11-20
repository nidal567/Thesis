import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import numpy as np

# Converting CartPoleEnv to SAC Implementation
class ContinuousCartPoleEnv(gym.Env):
    def __init__(self):
        super(ContinuousCartPoleEnv, self).__init__()
        
        # CartPole env init
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # Discrete to Continuous Action Space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, action):
        # Map continuous action (-1, 1) to discrete (0 or 1)
        discrete_action = 0 if action < 0 else 1
        obs, reward, done, truncated, info = self.env.step(discrete_action)
        
        # Modify the reward to encourage balance: give extra reward for smaller pole angles
        pole_angle = obs[2]
        reward = float(reward + (1.0 - abs(pole_angle)))  # Ensure reward is a float
        
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        self.env.render()

    def close(self):
        self.env.close()

# Continuous Env
env = ContinuousCartPoleEnv()

check_env(env) # Sanity check for continuous env compatibility with gymnasium

# Initialize the SAC model with a continuous env
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4, 
    ent_coef="auto_0.1",  # Higher entropy for more exploration initially
    batch_size=64,        # Small batch size to help it learn faster in a simple env
    gamma=0.99            # Discount factor close to 1 for longer-term reward focus
)

# Define the total training steps
total_timesteps = 20000
# Define interval for periodic evaluation
evaluation_interval = 1000
num_eval_episodes = 10

# Open time log file for recording episode durations
with open("time_log.txt", "w") as log_file:
    log_file.write("Training Step, Episode, Duration (Timesteps), Total Reward\n")  # Write header

    # Training loop with periodic evaluations
    for step in range(0, total_timesteps, evaluation_interval):
        # Train the model for a defined interval
        model.learn(total_timesteps=evaluation_interval, reset_num_timesteps=False)
        
        # Evaluate the model for `num_eval_episodes` episodes
        for episode in range(1, num_eval_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            duration = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                duration += 1  # Increment timestep counter for each step
                env.render()  # Render at each step for visualization
                
                if done or truncated:
                    print(f"Training Step: {step+evaluation_interval}, Episode: {episode}, Duration: {duration} timesteps, Total Reward: {total_reward}")
                    # Log the training step, episode duration, and reward to the file
                    log_file.write(f"{step+evaluation_interval}, {episode}, {duration}, {total_reward}\n")
                    log_file.flush()  # Write data immediately to the file
                    break

env.close() # Close environment once test is complete
