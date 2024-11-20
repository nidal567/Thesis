import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
import numpy as np

# Define the number of actions and states
num_actions = 5  # Replace with your actual number of actions
num_states = 10  # Replace with your actual number of states
initial_state = np.zeros(num_states)  # Example initial state

class WirelessNetworkEnv(gym.Env):
    def __init__(self):
        super(WirelessNetworkEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_states,), dtype=np.float32)

        # Initialize the state
        self.state = initial_state

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility
        self.state = initial_state  # Reset to the initial state
        return self.state, {}  # Return the initial state and an empty info dictionary

    def some_simulation_function(self, action):
        # Define how the state updates based on the action taken
        next_state = np.random.rand(num_states)  # Replace with your logic for the next state
        reward = np.random.rand()  # Replace with your logic for calculating the reward
        done = np.random.rand() > 0.95  # Randomly determine if the episode is done
        truncated = False  # For simplicity, we'll assume it isn't truncated
        info = {}  # Any additional info to return
        return next_state, reward, done, truncated, info  # Return all values

    def step(self, action):
        # Apply action, update state and calculate reward
        next_state, reward, done, truncated, info = self.some_simulation_function(action)
        self.state = next_state  # Update the current state
        return next_state, reward, done, truncated, info  # Return all values

# Option 1: Register the environment
gym.register(
    id='WirelessNetwork-v0',  # Unique identifier for the environment
    entry_point='__main__:WirelessNetworkEnv'  # Ensure that the path is correct
)

# Create the environment instance using gym.make
env = gym.make('WirelessNetwork-v0')  # Use this if you registered the environment

# Initialize the SAC model
model = SAC('MlpPolicy', env, verbose=1)

# Create a file to log the data
with open("training_log.txt", "w") as log_file:  # Using with statement for better file handling
    log_file.write("Episode, Total Reward, State\n")  # Write the header

    # Train the model
    model.learn(total_timesteps=10000)

    # Evaluate the trained model
    for episode in range(1):  # Number of episodes
        obs, _ = env.reset()  # Reset the environment
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)  # Added underscore for action distribution
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Log data
            log_entry = f"{episode+1}, {total_reward}, {obs.tolist()}\n"
            print(f"Logging data for Episode: {episode+1}, Total Reward: {total_reward}, State: {obs.tolist()}")
            log_file.write(log_entry)  # Write to the log file
            log_file.flush()  # Ensure data is written immediately
            
            # Print confirmation that data has been written
            print("Data written to log file.")

# Close the environment after use
env.close()
