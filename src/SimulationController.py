# SimulationController.py

import random
import numpy as np

from .MineSimulator import MineSimulator
from .DStarLite import DStarLite
from .constants import MOVE_TO_ACTION_MAP

# ===================================================================
# --- "Bridge" Logic: Functions that connect the Simulator and Planner ---
# These functions translate the rich state of the simulator into the simple
# numerical data that the pathfinder needs to operate.
# ===================================================================

def predict_battery(simulator, sensor_pos):
    """
    Placeholder for the future PPO-LSTM model.
    For now, it acts as a simple bridge, returning the current, true battery
    level directly from the simulator. When you train your model, you will

    replace the body of this function with a call to `model.predict()`.
    """
    return simulator.sensor_batteries.get(sensor_pos, 0.0)

def cost_function_for_dstar(simulator, pos_xy):
    """
    This is the most critical function for integration. It's the dynamic
    cost function that D* Lite will call for any given cell to determine its
    traversal cost.

    Args:
        simulator (MineSimulator): The instance of the live simulation.
        pos_xy (tuple): The (x, y) or (col, row) coordinate from D* Lite.
    
    Returns:
        float: The calculated traversal cost for that cell.
    """
    # D* Lite uses (x, y) coordinates, but the simulator uses (row, col). Convert them.
    pos_rc = (pos_xy[1], pos_xy[0])
    
    # In your model, the cost of a cell is determined by its closest sensor.
    if not simulator.sensor_positions:
        return 0.0 # No sensors, no extra cost
        
    # Find the closest sensor to the given position
    sensor_positions_rc = np.array(list(simulator.sensor_positions))
    distances = np.linalg.norm(sensor_positions_rc - np.array(pos_rc), axis=1)
    closest_sensor_pos = simulator.sensor_positions[np.argmin(distances)]

    # Get the (predicted) battery level for that sensor.
    battery_level = predict_battery(simulator, closest_sensor_pos)
    
    # Convert battery level into a cost. Lower battery = higher cost.
    # This is a critical part you will tune later.
    cost = 0.0
    if battery_level <= 10:
        cost = 200.0 # Extremely high cost for near-dead sensors
    elif battery_level <= 30:
        cost = 50.0  # High cost for low-battery sensors
    elif battery_level <= 60:
        cost = 10.0  # Moderate cost to discourage but not forbid
        
    return cost

def get_action_from_move(current_pos_rc, next_pos_rc):
    """
    Converts a move from a start coordinate to a destination coordinate
    into a simulator action ID (0-7).
    """
    # Calculate the vector of the move, e.g., (-1, 0) for UP
    move_vector = (next_pos_rc[0] - current_pos_rc[0], next_pos_rc[1] - current_pos_rc[1])
    # Look up the corresponding action ID in our constant map
    return MOVE_TO_ACTION_MAP.get(move_vector)

# ===================================================================
# --- The Controller Class ---
# ===================================================================

class SimulationController:
    """
    Acts as the Controller in an MVC-like pattern. It handles the main loop
    and orchestrates the interaction between the MineSimulator (Model)
    and the MineRenderer (View).
    """
    def __init__(self, grid_file: str, n_miners: int, render: bool = True):
        """
        Initializes the entire simulation system.
        """
        print("--- Initializing Simulation Controller ---")
        render_mode = "human" if render else None
        
        # The Controller OWNS the Model (Simulator)
        self.simulator = MineSimulator(grid_file=grid_file, n_miners=n_miners, render_mode=render_mode)
        
        # The Controller OWNS the Planner and tracks its state
        self.pathfinder = None
        self.dstar_path = [] # The FUTURE path calculated by D* Lite
        self.path_history = [] # The PAST (traveled) path of the miner
        self.is_running = False

        # Set up the pathfinder based on the simulator's initial state
        self._setup_pathfinder()

    def _setup_pathfinder(self):
        """Initializes the D* Lite planner and resets the path history."""
        initial_state = self.simulator.reset()
        
        # The path history starts with just the initial spawn point.
        self.path_history = [initial_state['guided_miner_pos']]

        # Create a simplified 0/1 grid for D* Lite's static obstacle map
        dstar_grid = np.zeros((self.simulator.n_rows, self.simulator.n_cols), dtype=int)
        for r, c in self.simulator.impassable_positions:
             if self.simulator.in_bounds((r,c)):
                dstar_grid[r, c] = 1

        # Convert coordinates from simulator's (row, col) to D* Lite's (x, y)
        start_pos_xy = (initial_state['guided_miner_pos'][1], initial_state['guided_miner_pos'][0])
        goals_xy = [(pos[1], pos[0]) for pos in initial_state['goal_positions']]
        
        # Create the D* Lite instance, passing it a lambda for the dynamic cost function.
        self.pathfinder = DStarLite(dstar_grid.tolist(), start_pos_xy, goals_xy,
                                    lambda pos: cost_function_for_dstar(self.simulator, pos))
        
        print("Computing initial path...")
        self.pathfinder._compute_shortest_path()
        
    def run(self):
        """The main execution loop of the simulation."""
        self.is_running = True
        while self.is_running:
            self.update_step()
        self.shutdown()

    def update_step(self):
        """Executes a single, complete step of the simulation logic."""
        # --- 1. Plan based on the START of the step ---
        # Get the path from the miner's current position.
        current_pos_rc = self.simulator.guided_miner_pos
        path_before_move = self.pathfinder.get_path_to_goal()
        action_to_take = None
        
        # --- 2. Decide on the action ---
        if path_before_move and len(path_before_move) > 1:
            next_pos_xy = path_before_move[1]
            next_pos_rc = (next_pos_xy[1], next_pos_xy[0])
            action_to_take = get_action_from_move(current_pos_rc, next_pos_rc)
        elif path_before_move and current_pos_rc in self.simulator.goal_positions:
            print("Goal Reached!")
            self.is_running = False
            return

        # --- 3. Act: Update the Model's state ---
        new_state = self.simulator.step(guided_miner_action=action_to_take)
        self.path_history.append(new_state['guided_miner_pos'])

        # --- 4. Update Planner with the new state ---
        new_pos_xy = (new_state['guided_miner_pos'][1], new_state['guided_miner_pos'][0])
        self.pathfinder.move_and_replan(new_pos_xy)
        sensor_positions_xy = [(pos[1], pos[0]) for pos in self.simulator.sensor_positions]
        self.pathfinder.update_costs(sensor_positions_xy)

        # --- 5. Get the NEW Path for Rendering ---
        # THE FIX: After the planner has been updated with the new position,
        # we ask it for the path again. This path will correctly start
        # from the miner's CURRENT position.
        path_for_render = self.pathfinder.get_path_to_goal()

        # --- 6. Render the final state of the world for this frame ---
        if self.simulator.render_mode == "human":
            render_status = self.simulator.render(
                show_miners=False,
                dstar_path=path_for_render, # Pass the NEW path
                path_history=self.path_history
            )
            if not render_status:
                self.is_running = False

    def shutdown(self):
        """Safely closes all components."""
        print("--- Simulation Finished ---")
        self.simulator.close()
