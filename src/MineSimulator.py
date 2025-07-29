# MineSimulator.py

import os
import random
import numpy as np

from . import grid_gen
from .MineRenderer import *
from .constants import (
    EMPTY_ID, OBSTACLE_ID, SENSOR_ID, BASE_STATION_ID, GOAL_ID,
    CHAR_TO_INT_MAP, DIRECTION_MAP, FIXED_GRID_DIR
)
from .sensor import *
from .utils import *

class MineSimulator:
    def __init__(self, grid_file: str, n_miners: int = 12, render_mode: str = None):
        self.agent_pos = None
        self.n_miners = n_miners
        self.miners = []
        self._init_grid_from_file(grid_file)
        self.render_mode = render_mode
        self.renderer = None

    def _init_grid_from_file(self, grid_file):
        grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
        char_grid, _, goals, sensors, base_stations = grid_gen.load_grid(grid_path)
        
        # Use the centralized map to convert the character grid to an integer grid
        int_grid = np.full(char_grid.shape, EMPTY_ID, dtype=int)
        for char, value in CHAR_TO_INT_MAP.items():
            int_grid[char_grid == char] = value
            
        self.static_grid = int_grid
        self.n_rows, self.n_cols = self.static_grid.shape
        self.goal_positions = tuple(goals)
        self.base_station_positions = tuple(base_stations)
        self.sensor_positions = tuple(sensors.keys())
        self.original_sensor_batteries = sensors
        self.n_sensors = len(self.sensor_positions)
        self.sensor_batteries = {}
        
        # Make interactive locations empty in the static grid
        for r, c in self.sensor_positions: self.static_grid[r, c] = EMPTY_ID
        for r, c in self.base_station_positions: self.static_grid[r, c] = EMPTY_ID

    def reset(self):
        self.sensor_batteries = self.original_sensor_batteries.copy()
        self.agent_pos = None
        while self.agent_pos is None:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.static_grid[r, c] == EMPTY_ID: self.agent_pos = (r, c)
        self.miners = []
        while len(self.miners) < self.n_miners:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.static_grid[r, c] == EMPTY_ID and (r, c) not in self.miners and (r,c) != self.agent_pos:
                self.miners.append((r, c))
        return self.get_state_snapshot()

    def step(self, agent_action=None):
        """
        Advances the simulation: moves entities and depletes sensor batteries.
        """
        # === Part 1: Move all entities to their new positions ===
        if agent_action is not None:
            move = DIRECTION_MAP[agent_action]
            new_pos = (self.agent_pos[0] + move[0], self.agent_pos[1] + move[1])
            if (0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols and self.static_grid[new_pos] == EMPTY_ID):
                self.agent_pos = new_pos
        
        self._move_miners_randomly()

        # === Part 2: Calculate battery depletion based on new positions ===
        # A dictionary mapping each sensor to a list of movers it services
        connections = {s_pos: [] for s_pos in self.sensor_positions}
        all_movers = self.miners + [self.agent_pos]

        if self.sensor_positions:
            # Determine which sensor each mover connects to
            for mover_pos in all_movers:
                closest_sensor = self._get_closest_sensor(mover_pos)
                if closest_sensor:
                    connections[closest_sensor].append(mover_pos)
        
        # Calculate battery drain for each sensor and update its state
        for sensor_pos, connected_movers in connections.items():
            energy_used = calculate_total_drain_on_sensor(
                sensor_pos,
                connected_movers,
                self.base_station_positions
            )
            
            battery_loss_percent = (energy_used / BATTERY_CAPACITY_JOULES) * 100
            current_battery = self.sensor_batteries[sensor_pos]
            self.sensor_batteries[sensor_pos] = max(0.0, current_battery - battery_loss_percent)

        # === Part 3: Return the final state of the world for this step ===
        return self.get_state_snapshot()

     def _get_closest_sensor(self, pos):
        """Finds the sensor closest to a position, now using the utility function."""
        if not self.sensor_positions: return None
        return min(self.sensor_positions, key=lambda s_pos: euclidean_distance(pos, s_pos))
    
    def get_state_snapshot(self):
        return {
            "agent_pos": self.agent_pos, "sensor_batteries": self.sensor_batteries.copy(),
            "miner_positions": self.miners.copy(), "base_station_positions": self.base_station_positions,
            "goal_positions": self.goal_positions,
        }

    def _move_miners_randomly(self):
        updated_miners = []
        for r, c in self.miners:
            move = random.choice(list(DIRECTION_MAP.values()))
            new_pos = (r + move[0], c + move[1])
            if (0 <= new_pos[0] < self.n_rows and 0 <= new_pos[1] < self.n_cols and self.static_grid[new_pos] == EMPTY_ID):
                updated_miners.append(new_pos)
            else:
                updated_miners.append((r, c))
        self.miners = updated_miners

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = MineRenderer(self.n_rows, self.n_cols)
            world_state = self.get_state_snapshot()
            return self.renderer.render(self.static_grid, world_state)
        return True

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
