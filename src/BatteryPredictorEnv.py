# BatteryPredictorEnv.py

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

from .MineSimulator import MineSimulator

class BatteryPredictorEnv(gym.Env):
    """
    A custom Gymnasium environment for training an agent to PREDICT battery levels.
    This environment conforms to the Stable-Baselines3 API and implements the MDP formulation similar to the one from Manish's paper

    - Action: The agent's prediction for the next battery level of all sensors.
    - Observation: The current state of all sensors (battery, miners, last prediction error).
    - Reward: How close the prediction was to the actual next-state battery level.
    """
    def __init__(self, grid_file: str, n_miners: int):
        super().__init__()
        
        # The environment contains an instance of the underlying physics simulation
        # to get the "ground truth" of what happens in the world.
        self.simulator = MineSimulator(grid_file=grid_file, n_miners=n_miners)
        self.num_sensors = self.simulator.n_sensors
        self.max_episode_steps = 1000 # End an episode after this many steps to avoid infinite loops

        # --- Define Action and Observation Spaces (crucial for SB3) ---

        # ACTION SPACE: A continuous prediction for each sensor's next battery level.
        # The values are normalized (0 to 1 represents 0% to 100%).
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_sensors,), dtype=np.float32)

        # OBSERVATION SPACE: The agent's perception of the world, based on Eq. 6 in the paper.
        # For each sensor, the agent sees 3 features:
        # 1. Current Battery Level (normalized 0-1)
        # 2. Number of Miners connected (normalized 0-1)
        # 3. Prediction Error from the last step (Δm, normalized -1 to 1)
        # We flatten this into a single vector of size num_sensors * 3.
        num_features = 3
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.num_sensors * num_features,), dtype=np.float32
        )

    def _get_observation(self, simulator_state):
        """Constructs the observation vector from the simulator's raw state."""
        # This carefully follows the paper's state space definition.
        
        # Get actual battery levels and normalize them
        actual_batteries_sorted = [simulator_state["sensor_batteries"][pos] for pos in self.simulator.sensor_positions]
        batteries_norm = np.array(actual_batteries_sorted) / 100.0
        
        # Calculate the number of miners connected to each sensor
        connections = self._get_connections(simulator_state)
        connections_sorted = [connections[pos] for pos in self.simulator.sensor_positions]
        miners_norm = np.array(connections_sorted) / self.simulator.n_miners
        
        # Prediction error: Δm = (actual_t - predicted_{t-1})
        error_norm = batteries_norm - self.last_predictions_norm

        # Stack and flatten into the final observation vector
        obs = np.stack([batteries_norm, miners_norm, error_norm], axis=1).flatten()
        return obs.astype(np.float32)

    def _get_connections(self, simulator_state):
        """Helper to count miners connected to each sensor."""
        connections = {s_pos: 0 for s_pos in self.simulator.sensor_positions}
        all_movers = simulator_state['miner_positions'] + [simulator_state['guided_miner_pos']]
        if self.simulator.sensor_positions:
            for mover_pos in all_movers:
                # This assumes a mover connects to its single closest sensor
                closest_sensor = min(self.simulator.sensor_positions, key=lambda s_pos: np.linalg.norm(np.array(mover_pos) - np.array(s_pos)))
                connections[closest_sensor] += 1
        return connections

    def reset(self, seed=None, options=None):
        """Called at the beginning of each training episode."""
        super().reset(seed=seed)
        
        initial_state = self.simulator.reset()
        self.current_step = 0
        
        # At the very start, we assume a "perfect" first prediction, so the initial error is 0.
        initial_batteries_sorted = [initial_state["sensor_batteries"][pos] for pos in self.simulator.sensor_positions]
        self.last_predictions_norm = np.array(initial_batteries_sorted) / 100.0
        
        observation = self._get_observation(initial_state)
        info = {}
        return observation, info

    def step(self, action):
        """
        Processes one step of the training loop. The agent provides an `action` (its prediction),
        and this function returns the result of that action.
        """
        # The 'action' from the agent is its array of normalized predictions.
        predicted_batteries_norm = action

        # --- Run the real physics simulation for one step ---
        # The guided miner moves randomly during training to explore different states.
        true_next_state = self.simulator.step(guided_miner_action=random.randint(0,7))
        true_next_batteries_sorted = [true_next_state["sensor_batteries"][pos] for pos in self.simulator.sensor_positions]
        true_next_batteries_norm = np.array(true_next_batteries_sorted) / 100.0

        # --- Calculate Reward (based on Eq. 8 in the paper) ---
        # The reward is higher if the prediction is closer to the actual result.
        # R = exp(-|B_actual - B_predicted| / C)
        errors = np.abs(true_next_batteries_norm - predicted_batteries_norm)
        # We average the reward across all sensors for a stable learning signal.
        reward = np.mean(np.exp(-errors / 0.1)) # C=0.1 is a tuning parameter from the paper
        
        # --- Check for episode termination ---
        self.current_step += 1
        # Episode ends if any sensor's battery dies (a failure state)
        terminated = np.any(np.array(true_next_batteries_sorted) <= 0)
        # Or if the episode runs for too long
        truncated = self.current_step >= self.max_episode_steps
        
        # --- Prepare the next observation for the agent ---
        # The prediction just made becomes the "last prediction" for the next step's error calculation.
        self.last_predictions_norm = predicted_batteries_norm
        observation = self._get_observation(true_next_state)
        
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources, like the Pygame window if it was opened."""
        self.simulator.close()
