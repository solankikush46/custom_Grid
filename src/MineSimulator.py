# MineSimulator.py

import os
import random
import numpy as np

from . import grid_gen
from .MineRenderer import MineRenderer
from .sensor import update_all_sensor_batteries
from .constants import (
    EMPTY_ID, OBSTACLE_ID,
    CHAR_TO_INT_MAP, DIRECTION_MAP, FIXED_GRID_DIR
)
from .utils import *

def get_closest_sensor(cell, sensor_positions):
    cr, cc = cell
    best = None
    best_d = None

    for sr, sc in sensor_positions:
        d = max(abs(sr - cr), abs(sc - cc))
        if best is None or d < best_d:
            best   = (sr, sc)
            best_d = d

    #print(f"[DEBUG get_closest_sensor] cell={cell} → sensor={best} (cheb_d={best_d})")
    
    return best

##==============================================================
## MineSimulator
##==============================================================
class MineSimulator:
    """
    Manages the state and physics of the mine simulation.
    Its core responsibilities are:
    - Loading the world layout.
    - Managing the state of all entities (guided_miner, miners, sensors).
    - Advancing the simulation state by one step according to game rules and physics.
    """
    def __init__(self, grid_file: str, n_miners: int = 12, render_mode: str = None, show_predicted: bool = False):
        """
        Initializes the Mine Simulator.

        Args:
            grid_file (str): The name of the .txt file in the fixed grids directory to load.
            n_miners (int): The number of autonomous miners to simulate.
            render_mode (str, optional): If set to "human", a renderer will be created
                                         for visualization. Defaults to None (headless).
            show_predicted (bool): If True, overlay predicted battery levels per cell.
        """
        # --- Core Simulation Attributes ---
        self.guided_miner_pos = None  # Position (row, col) of the single controllable miner
        self.n_miners = n_miners
        self.miners = []  # List of positions for the autonomous miners

        # Whether the renderer should overlay predicted battery
        self.show_predicted = show_predicted

        # Initialize the grid layout, sensor positions, etc., from a file
        self._init_grid_from_file(grid_file)

        # --- Rendering Attributes ---
        self.render_mode = render_mode
        self.renderer = None  # The renderer object will be created on the first call to render()

    def _init_grid_from_file(self, grid_file):
        """
        Loads the world layout from a text file and initializes
        the simulation state. Relies on the enhanced grid_gen.load_grid
        to parse all entity positions.
        """
        # --- Start of Initialization ---
        #print("\n=======================================")
        #print("--- Initializing Grid from File ---")
        #print(f"[Grid Init] Attempting to load: {grid_file}")

        grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
        
        # --- Step 1: Load Raw Data ---
        # Unpack all the lists of entities parsed by the grid_gen module.
        char_grid, _, goals, sensors, base_stations, obstacles = grid_gen.load_grid(grid_path)
        
        #print("\n[Grid Init] Raw data loaded from grid_gen:")
        #print(f"  - Goals found:         {len(goals)}")
        #print(f"  - Goal Positions:      {goals}")
        #print(f"  - Sensors found:         {len(sensors)}")
        #print(f"  - Base Stations found: {len(base_stations)}")
        #print(f"  - Obstacles found:       {len(obstacles)}")

        # --- Step 2: Set Dimensions and Create Static Grid ---
        self.n_rows, self.n_cols = char_grid.shape
        #print(f"\n[Grid Init] Grid dimensions set to: {self.n_rows} rows x {self.n_cols} cols")
        
        # The static_grid only contains permanent, impassable terrain (e.g., '#').
        # It starts as an empty grid of the correct size.
        self.static_grid = np.full((self.n_rows, self.n_cols), EMPTY_ID, dtype=int)
        # We then iterate through the list of obstacle positions and "paint" them onto the grid.
        for r, c in obstacles:
            self.static_grid[r, c] = OBSTACLE_ID
        #print(f"[Grid Init] Built static_grid containing {len(obstacles)} terrain obstacles.")
        
        # --- Step 3: Store Entity Positions and Data ---
        self.goal_positions = tuple(goals)
        self.base_station_positions = tuple(base_stations)
        self.sensor_positions = tuple(sensors.keys())
        self.original_sensor_batteries = sensors
        self.n_sensors = len(self.sensor_positions)
        self.sensor_batteries = {}  # Populated in reset()

        # sensor → list of (x,y) cells
        self.sensor_cell_map = {pos: [] for pos in self.sensor_positions}
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                nearest = get_closest_sensor((r, c), self.sensor_positions)
                self.sensor_cell_map[nearest].append((c, r))

        # cell → sensor
        self.cell_to_sensor = {}
        for s_pos, cells in self.sensor_cell_map.items():
            for cell_xy in cells:
                self.cell_to_sensor[cell_xy] = s_pos
        
        # --- Step 4: Build impassable set ---
        self.impassable_positions = set(obstacles).union(self.sensor_positions, self.base_station_positions)

        self.free_cells = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.impassable_positions and not (r, c) in self.goal_positions
        ]
        
        #print(f"\n[Grid Init] Built impassable_positions set with {len(self.impassable_positions)} total items.")
        #print(f"  - Example impassable positions: {list(self.impassable_positions)[:5]}")
        #print("--- Grid Initialization Complete ---")
        #print("=======================================\n")
        
    def reset(self):
        """
        Resets the simulation to its initial state for a new episode.
        """
        # Randomize each sensor's battery between 0 and 100
        self.sensor_batteries = {
            pos: random.uniform(0, 100)
            for pos in self.original_sensor_batteries.keys()
        }
       
        # Place the guided miner in a random passable cell
        self.guided_miner_pos = None
        while self.guided_miner_pos is None:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            pos = (r, c)
            if self.is_passable(pos):
                self.guided_miner_pos = pos
        
        # Spawn autonomous miners without collisions
        self.miners = []
        occupied = {self.guided_miner_pos}
        while len(self.miners) < self.n_miners:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            pos = (r, c)
            if self.is_passable(pos) and pos not in occupied:
                self.miners.append(pos)
                occupied.add(pos)
                
        return self.get_state_snapshot()

    def step(self, guided_miner_action=None):
        """
        Advances the simulation by one discrete timestep.

        Args:
            guided_miner_action (int, optional): An integer action for the guided miner.
        """
        # 1. Move guided miner
        if guided_miner_action is not None:
            dr, dc = DIRECTION_MAP[guided_miner_action]
            new_pos = (self.guided_miner_pos[0] + dr, self.guided_miner_pos[1] + dc)
            if self.is_valid_guided_miner_move(new_pos):
                self.guided_miner_pos = new_pos

        # 2. Move autonomous miners
        self._move_miners_randomly()

        # 3. Update all sensor batteries based on proximity to any mover
        all_movers = self.miners + [self.guided_miner_pos]
        self.sensor_batteries = update_all_sensor_batteries(
            self.sensor_positions,
            self.sensor_batteries,
            all_movers,
            self.base_station_positions
        )
        
        # Return the new world state
        return self.get_state_snapshot()
    
    def get_state_snapshot(self):
        """
        Gathers and returns the complete state of the simulation world.
        """
        return {
            "guided_miner_pos":       self.guided_miner_pos,
            "sensor_batteries":       self.sensor_batteries.copy(),
            "miner_positions":        self.miners.copy(),
            "base_station_positions": self.base_station_positions,
            "goal_positions":         self.goal_positions,
        }

    ##===================================================================
    ## Movement Logic Helpers
    ##===================================================================
    def in_bounds(self, pos):
        """Checks if a position (row, col) is within grid bounds."""
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_passable(self, pos):
        """True if `pos` is not an obstacle, sensor, or base station."""
        return pos not in self.impassable_positions

    def is_valid_guided_miner_move(self, pos):
        """Guided miner may move anywhere passable (including miner‐occupied)."""
        return self.in_bounds(pos) and self.is_passable(pos)

    def is_valid_miner_move(self, pos, occupied):
        """Autonomous miners cannot collide with guided miner or other miners."""
        return self.is_valid_guided_miner_move(pos) and pos not in occupied

    def _move_miners_randomly(self):
        """
        Moves autonomous miners randomly, ensuring valid, non‐colliding moves.
        """
        shuffled = random.sample(self.miners, len(self.miners))
        occupied = {self.guided_miner_pos}
        new_positions = []
        for m in shuffled:
            dr, dc = random.choice(list(DIRECTION_MAP.values()))
            cand = (m[0] + dr, m[1] + dc)
            if self.is_valid_miner_move(cand, occupied):
                new_positions.append(cand)
                occupied.add(cand)
            else:
                new_positions.append(m)
                occupied.add(m)
        self.miners = new_positions

    ##==============================================================
    ## PyGame Rendering Delegation
    ##==============================================================
    def render(self,
               show_miners: bool = True,
               dstar_path: list = None,
               path_history: list = None,
               predicted_battery_map: np.ndarray = None):
        """
        High-level rendering API. Delegates to MineRenderer, passing along
        an optional predicted_battery_map overlay.
        """
        if self.render_mode == "human":
            if self.renderer is None:
                # pass along the show_predicted flag
                self.renderer = MineRenderer(self.n_rows,
                                             self.n_cols,
                                             show_predicted=self.show_predicted)
            state = self.get_state_snapshot()
            return self.renderer.render(
                self.static_grid,
                state,
                show_miners,
                dstar_path,
                path_history,
                predicted_battery_map  # now accepted
            )
        return True

    def close(self):
        """Safely shuts down Pygame if initialized."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
