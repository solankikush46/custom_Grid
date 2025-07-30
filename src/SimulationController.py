# SimulationController.py

import os
import numpy as np

from .MineSimulator import MineSimulator
from .DStarLite import DStarLite
from .constants import MOVE_TO_ACTION_MAP, SAVE_DIR

class SimulationController:
    """
    Controller integrating MineSimulator and a standard D* Lite planner.
    The pathfinding cost is based on the current state of sensor batteries.
    This controller is designed to run simulations back-to-back until shutdown.
    """
    def __init__(
        self,
        experiment_folder: str,
        render: bool = True
    ):
        """
        Performs one-time setup of the simulation environment.

        Args:
            experiment_folder: Name of the experiment folder under SAVE_DIR
                               (e.g., 'mine_20x20_5miners').
            render: Whether to render in human mode.
        """
        # --- This one-time setup is performed only when the class is created ---
        exp_name = experiment_folder
        parts = exp_name.split('_')
        if len(parts) < 3 or not parts[-1].endswith('miners'):
            raise ValueError(
                f"Experiment folder '{exp_name}' must be like 'mine_<WxH>_<N>miners'."
            )
        grid_file = f"{parts[0]}_{parts[1]}.txt"
        try:
            n_miners = int(parts[-1].replace('miners',''))
        except ValueError:
            raise ValueError(
                f"Cannot parse number of miners from '{parts[-1]}'."
            )

        # Initialize the simulator (this happens only once)
        render_mode = 'human' if render else None
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode
        )

        # Initialize state variables
        self.pathfinder = None
        self.path_history = []
        self.is_running = False # Controls the loop for a single simulation
        self.should_shutdown = False # Controls the outer loop for continuous runs

    def _dstar_cost(self, pos_xy):
        """
        Cost callback for D* Lite, using current sensor battery levels.
        A high cost is assigned to moving near a sensor with low battery.
        """
        if not self.simulator.sensor_positions:
            return 0.0
        
        # Convert position to (row, col) for distance calculation
        rc = (pos_xy[1], pos_xy[0])
        sensors = list(self.simulator.sensor_positions)
        
        # Find the nearest sensor to the given position
        distances = np.linalg.norm(np.array(sensors) - np.array(rc), axis=1)
        nearest_sensor = sensors[int(np.argmin(distances))]
        
        # Get the current battery level of this sensor
        batt = self.simulator.sensor_batteries[nearest_sensor]
        
        # Map battery level to a cost
        if batt <= 10:   return 200.0  # Very high cost for critical battery
        if batt <= 30:   return  50.0  # High cost for low battery
        if batt <= 60:   return  10.0  # Moderate cost for medium battery
        return 0.0 # No cost for healthy battery

    def _setup_pathfinder(self):
        """
        Resets the simulation environment and re-initializes D* Lite for a new run.
        """
        print("\n[SETUP] Resetting environment for new simulation run...")
        state = self.simulator.reset()
        self.path_history = [state['guided_miner_pos']]

        # Create a static grid representing impassable obstacles
        grid = np.zeros((self.simulator.n_rows, self.simulator.n_cols), dtype=int)
        for r, c in self.simulator.impassable_positions:
            if self.simulator.in_bounds((r,c)):
                grid[r, c] = 1

        start = (state['guided_miner_pos'][1], state['guided_miner_pos'][0])
        goals = [(y, x) for x, y in state['goal_positions']]

        # Initialize D* Lite with the grid and the cost function
        self.pathfinder = DStarLite(
            grid.tolist(),
            start,
            goals,
            lambda pos: self._dstar_cost(pos)
        )
        print("[INFO] Computing initial path...")
        self.pathfinder._compute_shortest_path()

    def run(self):
        """
        Main loop that continuously resets and runs simulations until the
        render window is closed or the program is interrupted.
        """
        while not self.should_shutdown:
            # Setup a new simulation run (resets miner, goals, etc.)
            self._setup_pathfinder()
            self.is_running = True
            
            # Inner loop for the current simulation instance
            while self.is_running:
                self.update_step()

        # This is called only once after the main loop has been exited
        self.shutdown()

    def update_step(self):
        """Perform one step: step simulation, replan path, and render."""
        pos = self.simulator.guided_miner_pos
        path = self.pathfinder.get_path_to_goal()

        # Determine the next action from the path
        if path and len(path) > 1:
            nxt = path[1]
            move = (nxt[1] - pos[0], nxt[0] - pos[1])
            act = MOVE_TO_ACTION_MAP.get(move)
        elif pos in self.simulator.goal_positions:
            print("[SUCCESS] Goal reached.")
            self.is_running = False # End current run, the outer loop will start a new one
            return
        else:
            # No path found or already at goal, stay put
            act = None

        # Step the simulator
        new_state = self.simulator.step(guided_miner_action=act)
        self.path_history.append(new_state['guided_miner_pos'])

        # Replan the path based on the new miner position and updated environment
        current_pos_rc = (new_state['guided_miner_pos'][1], new_state['guided_miner_pos'][0])
        self.pathfinder.move_and_replan(current_pos_rc)
        
        # Inform D* Lite of any potential cost changes around the sensors
        sensor_xy = [(y, x) for x, y in self.simulator.sensor_positions]
        self.pathfinder.update_costs(sensor_xy)

        # Render the current state
        if self.simulator.render_mode == 'human':
            ok = self.simulator.render(
                show_miners=False,
                dstar_path=self.pathfinder.get_path_to_goal(),
                path_history=self.path_history
            )
            if not ok:
                # User closed the window, trigger a full shutdown.
                print("[INFO] Render window closed by user.")
                self.is_running = False
                self.should_shutdown = True

    def shutdown(self):
        """Clean up resources."""
        print("--- Shutting Down All Simulations ---")
        self.simulator.close()
