# SimulationController.py

import os
import numpy as np
from .MineSimulator import MineSimulator
from .DStarLite import DStarLite
from .constants import MOVE_TO_ACTION_MAP, SAVE_DIR

class SimulationController:
    """
    Controller integrating MineSimulator with the standalone D* Lite planner.
    Uses battery level costs via an external cost_function.
    """

    def __init__(self, experiment_folder: str, render: bool = True):
        parts = experiment_folder.split('_')
        if len(parts) < 3 or not parts[-1].endswith('miners'):
            raise ValueError(f"Experiment folder '{experiment_folder}' must be 'mine_<WxH>_<N>miners'")
        grid_file = f"{parts[0]}_{parts[1]}.txt"
        try:
            n_miners = int(parts[-1].replace('miners',''))
        except ValueError:
            raise ValueError(f"Cannot parse number of miners from '{parts[-1]}'")

        render_mode = 'human' if render else None
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode
        )

        self.pathfinder: DStarLite = None
        self.path_history = []
        self.is_running = False
        self.should_shutdown = False

    def _battery_cost(self, cell_xy):
        """
        Battery-based cost callback: maps grid-cell (x,y) → float.
        Simulator stores battery by (row,col), so swap:
        """
        row, col = cell_xy[1], cell_xy[0]
        # Find nearest sensor
        sensors = list(self.simulator.sensor_positions)
        dists   = np.linalg.norm(np.array(sensors) - np.array((row, col)), axis=1)
        nearest = sensors[int(np.argmin(dists))]
        batt    = self.simulator.sensor_batteries[nearest]
        # Map battery to additional cost
        if batt <= 10:   return 200.0
        if batt <= 30:   return  50.0
        if batt <= 60:   return  10.0
        return 0.0

    def _setup_pathfinder(self):
        print("\n[SETUP] Starting new simulation run…")
        state = self.simulator.reset()
        self.path_history = [state['guided_miner_pos']]

        static_obs = set(self.simulator.impassable_positions)
        W, H = self.simulator.n_cols, self.simulator.n_rows

        # Convert guided_miner_pos from (row, col) to (x, y):
        start_r, start_c = state['guided_miner_pos']
        start = (start_c, start_r)

        # First goal (row,col) → (x,y)
        goal_r, goal_c = state['goal_positions'][0]
        goal = (goal_c, goal_r)

        # Initialize D* Lite with external cost function
        self.pathfinder = DStarLite(
            width=W,
            height=H,
            start=start,
            goal=goal,
            cost_function=self._battery_cost,
            known_obstacles={(c, r) for (r, c) in static_obs},
            heuristic=None
        )
        self.pathfinder.compute_shortest_path()

    def run(self):
        while not self.should_shutdown:
            self._setup_pathfinder()
            self.is_running = True
            while self.is_running:
                self.update_step()
        self.shutdown()

    def update_step(self):
        pos_r, pos_c = self.simulator.guided_miner_pos
        path = self.pathfinder.get_shortest_path() or []
        if not path:
            x = input("[PAUSE] Path empty")

        if len(path) > 1:
            nxt_x, nxt_y = path[1]
            move = (nxt_y - pos_r, nxt_x - pos_c)  # (Δrow, Δcol)
            act  = MOVE_TO_ACTION_MAP.get(move)
        elif self.simulator.guided_miner_pos in self.simulator.goal_positions:
            print("[SUCCESS] Reached goal.")
            self.is_running = False
            return
        else:
            act = None

        new_state = self.simulator.step(guided_miner_action=act)
        self.path_history.append(new_state['guided_miner_pos'])

        # Update start + km
        old = self.pathfinder.start
        nr, nc = new_state['guided_miner_pos']
        new = (nc, nr)
        if new != old:
            self.pathfinder.start = new
            self.pathfinder.km   += self.pathfinder.h(old, new)

        # Determine which grid cells changed cost.
        # (for now, we conservatively use every cell):
        all_sensor_positions_xy = [(pos[1], pos[0]) for pos in self.simulator.sensor_batteries.keys()]
        all_cells = [
            (c, r)
            for r in range(self.simulator.n_rows)
            for c in range(self.simulator.n_cols)
        ]

        # Notify D* Lite of cost changes:
        for cell in all_cells:
            for u in self.pathfinder.pred(cell) + [cell]:
                self.pathfinder.update_vertex(u)
        
        # Repair path
        self.pathfinder.compute_shortest_path()

        # Render
        if self.simulator.render_mode == 'human':
            ok = self.simulator.render(
                show_miners=False,
                dstar_path=self.pathfinder.get_shortest_path(),
                path_history=self.path_history
            )
            if not ok:
                print("[INFO] Window closed by user.")
                self.is_running       = False
                self.should_shutdown = True

    def shutdown(self):
        print("--- Shutting down simulations ---")
        self.simulator.close()
