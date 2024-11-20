import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# Create the environment
env = gym.make("ALE/MsPacman-v5", render_mode="human")
env = VecTransposeImage(DummyVecEnv([lambda: env]))  # Wrap the environment

# Initialize the PPO model
model = PPO('CnnPolicy', env, verbose=1)

# Training loop
total_timesteps = 100000  # Adjust based on your needs
model.learn(total_timesteps=total_timesteps)

# Evaluate the trained model
obs, info = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)  # Get action from the trained model
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()

env.close()