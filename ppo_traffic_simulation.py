"""
PPO Traffic Light Control System

This module implements a Proximal Policy Optimization (PPO) algorithm for 
traffic light control using a custom Gymnasium environment that interfaces 
with NetLogo simulation.
"""

import numpy as np
import random
import os
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import pynetlogo


class TrafficSimulation(Env):
    """
    Custom Gymnasium environment for traffic light control simulation.
    
    This environment interfaces with a NetLogo traffic simulation model
    and uses PPO to learn optimal traffic light timing strategies.
    """
    
    def __init__(self):
        super(TrafficSimulation, self).__init__()

        # Define the observation space as a Box space with densities
        self.observation_space = Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)

        # Define the action space as a MultiDiscrete
        self.action_space = MultiDiscrete([
            7,  # com: 7 possible combinations (0 to 6, mapped to 1-7)
            36, 36, 36, 36  # side-timers: 4 values each ranging from 5 to 40
        ])
        
        # Episode length and state initialization
        self.episode_length = 15
        self.current_step = 0

        # Initialize NetLogo connection
        self.netlogo = pynetlogo.NetLogoLink(
            jvm_path=r"C:\Program Files\Java\jdk-19\bin\server\jvm.dll",
            gui=True,
        )

        self.netlogo.load_model(r'4Way-Junction-Traffic-Simulation-SriLanka.nlogo')
        self.netlogo.command('setup')

        # Initialize previous density standard deviation for reward calculation
        self.previous_density_stdev = 0

    def reset(self, seed=None, options=None):
        """Reset the environment state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.netlogo.command('setup')
        
        # Initialize observation with default densities as zeros
        observation = np.zeros(8, dtype=np.float32)
        self.previous_density_stdev = 0

        return observation, {}

    def step(self, action):
        """Perform one step in the environment."""
        # Decode the action
        selected_combination = int(action[0]) + 1
        green_light_durations = [int(g + 5) for g in action[1:]]
    
        # Update NetLogo with the selected combination and timers
        self.netlogo.command(f'set routes-combinations "com{selected_combination}"')
        self.netlogo.command(f'set side1 {green_light_durations[0]}')
        self.netlogo.command(f'set side2 {green_light_durations[1]}')
        self.netlogo.command(f'set side3 {green_light_durations[2]}')
        self.netlogo.command(f'set side4 {green_light_durations[3]}')
    
        # Run the NetLogo model until the cycle completes
        self.netlogo.command('go-cycle')
    
        # Get the new observation (densities)
        observation = self.get_current_densities()
    
        # Check if all observations are 0 (potential deadlock)
        if np.all(observation == 0):
            print("Deadlock detected: All densities are 0. Ending episode early.")
            reward = -10
            done = True
            truncated = True
            info = {"reason": "deadlock"}
            return observation, reward, done, truncated, info

        # Calculate the minimum density and set it as min-density in NetLogo
        min_density = max(np.min(observation), 0.8) 
        print(f"Setting min-density to {min_density}")
        self.netlogo.command(f'set min-density {min_density}')
    
        # Calculate reward based on density standard deviation
        new_density_stdev = np.std(observation)
        reward = 0
        if new_density_stdev < self.previous_density_stdev:
            reward = 1
        elif new_density_stdev > self.previous_density_stdev:
            reward = -1
    
        self.previous_density_stdev = new_density_stdev
    
        # Check if the episode is done
        self.current_step += 1
        done = self.current_step >= self.episode_length
    
        truncated = False
        info = {}
    
        return observation, reward, done, truncated, info

    def get_current_densities(self):
        """Get the current traffic densities from NetLogo."""
        densities_dict = self.check_density()
        densities = np.array(list(densities_dict.values()))
        return densities

    def check_density(self):
        """Calculate and return traffic densities based on the current route combination."""
        current_com = self.netlogo.report('routes-combinations')

        # Define the routes combination dictionary
        routes_combination = {
            "com1": ['R12', 'R34', 'R56', 'R78', None],
            "com2": ['R12', 'R37', 'R48', 'R56', None],
            "com3": ['R14', 'R26', 'R37', 'R58', None],
            "com4": ['R14', 'R27', 'R36', 'R58', None],
            "com5": ['R15', 'R26', 'R34', 'R78', None],
            "com6": ['R15', 'R26', 'R37', 'R48', None],
            "com7": ['R15', 'R27', 'R36', 'R48', None]
        }
        
        # Define the routes dictionary
        routes = {
            'R12': ['S1', 'S2', None],
            'R14': ['S1', 'S4', None],
            'R15': ['S1', 'S5', None],
            'R26': ['S2', 'S6', None],
            'R27': ['S2', 'S7', None],
            'R34': ['S3', 'S4', None],
            'R36': ['S3', 'S6', None],
            'R37': ['S3', 'S7', None],
            'R48': ['S4', 'S8', None],
            'R56': ['S5', 'S6', None],
            'R58': ['S5', 'S8', None],
            'R78': ['S7', 'S8', None]
        }

        # Select the specific combination from the dictionary
        selected_combination_routes = routes_combination.get(current_com, [])
        selected_combination_routes = [route for route in selected_combination_routes if route is not None]
        
        # Create a sides dictionary for the selected combination
        sides = {f'side{i+1}': route for i, route in enumerate(selected_combination_routes)}

        # Construct the modes dictionary using the routes dictionary
        modes = {}
        
        for side, route in sides.items():
            sensors = routes.get(route, [])
            sensors = [sensor for sensor in sensors if sensor is not None]
            modes[side] = [None, {sensor: None for sensor in sensors}]

        # Initialize densities dictionary for all sensors
        result = {'S1': None, 'S2': None, 'S3': None, 'S4': None, 
                  'S5': None, 'S6': None, 'S7': None, 'S8': None}

        def get_variable(s):
            return int(self.netlogo.report(f'{s}'))

        def v_count(name):
            result = self.netlogo.report(
                f'ifelse-value (any? lights with [name = "{name}"]) [ [cars-passed] of one-of lights with [name = "{name}"] ] [ 0 ]'
            )
            return int(result)

        # Calculate the densities based on the current mode settings
        for key, value in modes.items():
            modes[key][0] = get_variable(key)
            for sub_key, sub_value in value[1].items():
                vehicle_count = v_count(sub_key)
                
                if value[0] != 0:
                    result[sub_key] = round(vehicle_count / value[0], 2)
                else:
                    result[sub_key] = 0.0

        return result

    def render(self):
        """Visualization is handled by NetLogo."""
        pass

    def close(self):
        """Close the NetLogo connection."""
        self.netlogo.kill_workspace()


def train_ppo_model(timesteps=30000, log_dir='Training/Logs', model_path='Training/PPO_Traffic_Model'):
    """
    Train a PPO model for traffic light control.
    
    Args:
        timesteps (int): Number of training timesteps
        log_dir (str): Directory for tensorboard logs
        model_path (str): Path to save the trained model
    
    Returns:
        PPO: Trained PPO model
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize the environment
    env = TrafficSimulation()
    
    # Create the PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir
    )
    
    print(f"Starting training for {timesteps} timesteps...")
    # Train the model
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Close the environment
    env.close()
    
    return model


def test_ppo_model(model_path='Training/PPO_Traffic_Model', num_steps=100):
    """
    Test a trained PPO model.
    
    Args:
        model_path (str): Path to the saved model
        num_steps (int): Number of steps to run the simulation
    """
    # Initialize the environment
    env = TrafficSimulation()
    env = Monitor(env)
    
    # Load the saved model
    model = PPO.load(model_path, env=env)
    
    # Reset the environment
    observation, _ = env.reset()
    
    cumulative_reward = 0
    
    for step in range(num_steps):
        # Predict the action
        action, _ = model.predict(observation, deterministic=True)
        
        # Take the action
        observation, reward, done, truncated, info = env.step(action)
        
        cumulative_reward += reward

        print(f"Step {step + 1}:")
        print(f"Action Taken: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Cumulative Reward: {cumulative_reward}")
        print("-" * 30)
        
        if done or truncated:
            print(f"Episode finished. Cumulative Reward: {cumulative_reward}")
            observation, _ = env.reset()
            cumulative_reward = 0
    
    # Close the environment
    env.close()


def evaluate_ppo_model(model_path='Training/PPO_Traffic_Model', n_eval_episodes=10):
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path (str): Path to the saved model
        n_eval_episodes (int): Number of episodes for evaluation
    
    Returns:
        tuple: Mean reward and standard deviation
    """
    # Initialize the environment
    env = TrafficSimulation()
    
    # Load the saved model
    model = PPO.load(model_path, env=env)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    # Close the environment
    env.close()
    
    return mean_reward, std_reward


if __name__ == "__main__":
    # Example usage
    print("PPO Traffic Light Control System")
    print("=" * 40)
    
    # Train the model
    print("Training PPO model...")
    model = train_ppo_model(timesteps=30000)
    
    # Test the model
    print("\nTesting PPO model...")
    test_ppo_model(num_steps=50)
    
    # Evaluate the model
    print("\nEvaluating PPO model...")
    mean_reward, std_reward = evaluate_ppo_model(n_eval_episodes=10)