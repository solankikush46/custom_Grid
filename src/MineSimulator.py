# MineSimulator.py

import os
import random
import numpy as np

# --- Relative imports from other modules within the 'src' package ---
from . import grid_gen
from .MineRenderer import MineRenderer
from .sensor import update_all_sensor_batteries
from .constants import (
    EMPTY_ID, OBSTACLE_ID,
    CHAR_TO_INT_MAP, DIRECTION_MAP, FIXED_GRID_DIR
)

class MineSimulator:
    """
    Manages the state and physics of the mine simulation.
    Its core responsibilities are:
    - Loading the world layout.
    - Managing the state of all entities (guided_miner, miners, sensors).
    - Advancing the simulation state by one step according to game rules and physics.
    """
    def __init__(self, grid_file: str, n_miners: int = 12, render_mode: str = None):
        """
        Initializes the Mine Simulator.

        Args:
            grid_file (str): The name of the .txt file in the fixed grids directory to load.
            n_miners (int): The number of autonomous miners to simulate.
            render_mode (str, optional): If set to "human", a renderer will be created
                                         for visualization. Defaults to None (headless).
        """
        # --- Core Simulation Attributes ---
        self.guided_miner_pos = None  # Position (row, col) of the single controllable miner
        self.n_miners = n_miners
        self.miners = []  # List of positions for the autonomous miners

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
        print("\n=======================================")
        print("--- Initializing Grid from File ---")
        print(f"[Grid Init] Attempting to load: {grid_file}")

        grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
        
        # --- Step 1: Load Raw Data ---
        # Unpack all the lists of entities parsed by the grid_gen module.
        char_grid, _, goals, sensors, base_stations, obstacles = grid_gen.load_grid(grid_path)
        
        print("\n[Grid Init] Raw data loaded from grid_gen:")
        print(f"  - Goals found:         {len(goals)}")
        print(f"  - Goal Positions:      {goals}")
        print(f"  - Sensors found:         {len(sensors)}")
        print(f"  - Base Stations found: {len(base_stations)}")
        print(f"  - Obstacles found:       {len(obstacles)}")

        # --- Step 2: Set Dimensions and Create Static Grid ---
        self.n_rows, self.n_cols = char_grid.shape
        print(f"\n[Grid Init] Grid dimensions set to: {self.n_rows} rows x {self.n_cols} cols")
        
        # The static_grid only contains permanent, impassable terrain (e.g., '#').
        # It starts as an empty grid of the correct size.
        self.static_grid = np.full((self.n_rows, self.n_cols), EMPTY_ID, dtype=int)
        # We then iterate through the list of obstacle positions and "paint" them onto the grid.
        for r, c in obstacles:
            self.static_grid[r, c] = OBSTACLE_ID
        print(f"[Grid Init] Built static_grid containing {len(obstacles)} terrain obstacles.")
        
        # --- Step 3: Store Entity Positions and Data ---
        # Store the positions of all other entities as attributes of the simulator.
        self.goal_positions = tuple(goals)
        self.base_station_positions = tuple(base_stations)
        self.sensor_positions = tuple(sensors.keys())
        self.original_sensor_batteries = sensors
        self.n_sensors = len(self.sensor_positions)
        self.sensor_batteries = {} # This will be populated during reset()
        
        # --- Step 4: Build the Optimized Set for Collision Checking ---
        # The impassable_positions set is the single source of truth for movement validation.
        # It combines all types of "walls" into one highly efficient data structure.
        self.impassable_positions = set(obstacles).union(self.sensor_positions, self.base_station_positions)
        
        print(f"\n[Grid Init] Built impassable_positions set with {len(self.impassable_positions)} total items.")
        # Showing a small sample can be useful for debugging without flooding the console.
        print(f"  - Example impassable positions: {list(self.impassable_positions)[:5]}")
        
        print("--- Grid Initialization Complete ---")
        print("=======================================\n")
        
    def reset(self):
        """
        Resets the simulation to its initial state for a new episode.
        """
        self.sensor_batteries = {
            pos: random.uniform(0, 100)
            for pos in self.original_sensor_batteries.keys()
        }
       
        # Place the controllable guided_miner at a random valid spot
        self.guided_miner_pos = None
        while self.guided_miner_pos is None:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            pos = (r, c)
            if self.is_passable(pos):
                self.guided_miner_pos = pos
        
        # Place autonomous miners, ensuring no collisions
        self.miners = []
        occupied_for_spawn = {self.guided_miner_pos} # guided_miner's spot is already claimed
        while len(self.miners) < self.n_miners:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            pos = (r, c)
            if self.is_passable(pos) and pos not in occupied_for_spawn:
                self.miners.append(pos)
                occupied_for_spawn.add(pos)
                
        return self.get_state_snapshot()

    def step(self, guided_miner_action=None):
        """
        Advances the simulation by one discrete timestep.

        Args:
            guided_miner_action (int, optional): An integer action for the controllable miner.
        """
        # --- 1. MOVEMENT PHASE ---
        if guided_miner_action is not None:
            move = DIRECTION_MAP[guided_miner_action]
            new_pos = (self.guided_miner_pos[0] + move[0], self.guided_miner_pos[1] + move[1])
            if self.is_valid_guided_miner_move(new_pos):
                self.guided_miner_pos = new_pos
        self._move_miners_randomly()

        # --- 2. BATTERY DEPLETION PHASE ---
        all_movers = self.miners + [self.guided_miner_pos]
        self.sensor_batteries = update_all_sensor_batteries(
            self.sensor_positions, self.sensor_batteries, all_movers, self.base_station_positions
        )
        
        # --- 3. FINAL STATE ---
        return self.get_state_snapshot()
    
    def get_state_snapshot(self):
        """
        Gathers and returns the complete state of the simulation world.
        """
        return {
            "guided_miner_pos": self.guided_miner_pos,
            "sensor_batteries": self.sensor_batteries.copy(),
            "miner_positions": self.miners.copy(),
            "base_station_positions": self.base_station_positions,
            "goal_positions": self.goal_positions,
        }

    ##===================================================================
    ## Movement Logic Helpers
    ##===================================================================

    def in_bounds(self, pos):
        """Checks if a position (row, col) is within the grid boundaries."""
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_passable(self, pos):
        """
        Checks if a position is passable terrain (not an obstacle, sensor, or base station).
        """
        return pos not in self.impassable_positions

    def is_valid_guided_miner_move(self, pos):
        """Checks if a move is valid for the guided_miner (can move onto other miner cells)."""
        return self.in_bounds(pos) and self.is_passable(pos)

    def is_valid_miner_move(self, pos, occupied_cells):
        """Checks if a move is valid for an autonomous miner (cannot move onto occupied cells)."""
        return self.is_valid_guided_miner_move(pos) and pos not in occupied_cells

    def _move_miners_randomly(self):
        """
        Moves autonomous miners randomly, ensuring they only move to valid, unoccupied cells.
        """
        shuffled_miners = random.sample(self.miners, len(self.miners))
        occupied_next_turn = {self.guided_miner_pos}
        updated_miners = []
        
        for miner_pos in shuffled_miners:
            move = random.choice(list(DIRECTION_MAP.values()))
            new_pos = (miner_pos[0] + move[0], miner_pos[1] + move[1])
            
            if self.is_valid_miner_move(new_pos, occupied_next_turn):
                updated_miners.append(new_pos)
                occupied_next_turn.add(new_pos)
            else:
                updated_miners.append(miner_pos)
                occupied_next_turn.add(miner_pos)

        self.miners = updated_miners

    ##==============================================================
    ## PyGame Rendering Delegation
    ##==============================================================

    def render(self, show_miners=True, dstar_path=None, path_history=None):
        """
        High-level rendering API. Delegates the drawing to its internal renderer.
        """
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = MineRenderer(self.n_rows, self.n_cols)
            
            world_state = self.get_state_snapshot()
            return self.renderer.render(self.static_grid,
                                        world_state,
                                        show_miners,
                                        dstar_path,
                                        path_history)
        
        return True

    def close(self):
        """Safely closes the renderer and shuts down Pygame if it was initialized."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
