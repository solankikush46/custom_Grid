# SimulationController.py

import os
import numpy as np

from .MineSimulator import MineSimulator
from .DStarLite.DStarLite import DStarLite  # C++ pybind11 module
from .constants import MOVE_TO_ACTION_MAP

# evaluation metrics
# - path len
# - number of timesteps connected to depleted sensor
#   (guided miner's randomized pos should be in non-depleted zone)

def sensor_cost_tier(batt):
    if batt <= 5:    return 400.0
    if batt <= 10:   return 200.0
    if batt <= 30:   return 50.0
    return 0.0

class SimulationController:
    """
    Integrates MineSimulator with the C++ D* Lite planner.
    Uses precomputed cost map and incremental updates via updateStart and updateVertex.
    """
    def __init__(self, experiment_folder: str, render: bool = True):
        parts = experiment_folder.split('_')
        grid_file = f"{parts[0]}_{parts[1]}.txt"
        n_miners  = int(parts[-1].replace('miners',''))
        render_mode = 'human' if render else None

        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode
        )

        self.pathfinder = None
        self.path_history = []
        self.is_running     = False
        self.should_shutdown = False

        # Track last cost-tier per sensor
        self.sensor_previous_costs = {}

    def _setup_pathfinder(self):
        # 1) reset sim and record start
        state = self.simulator.reset()
        self.path_history = [ state['guided_miner_pos'] ]
        # 2) init previous tiers
        for s_pos, batt in self.simulator.sensor_batteries.items():
            self.sensor_previous_costs[s_pos] = sensor_cost_tier(batt)

        # 3) build cost map (H x W) row-major: cost_map[y][x]
        H, W = self.simulator.n_rows, self.simulator.n_cols
        self.cost_map = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                sensor = self.simulator.cell_to_sensor[(x, y)]
                batt   = self.simulator.sensor_batteries[sensor]
                self.cost_map[y, x] = sensor_cost_tier(batt)

        # 4) build static obstacle list
        static_obs = [(x, y) for (y, x) in self.simulator.impassable_positions]

        # 5) start in (x,y)
        sr, sc   = state['guided_miner_pos']
        start_x, start_y = sc, sr

        # 6) goals as list of (x,y)
        goals = [(c, r) for (r, c) in state['goal_positions']]

        # 7) init C++ D* Lite planner
        self.pathfinder = DStarLite(
            W, H,
            start_x, start_y,
            goals,
            self.cost_map,
            static_obs
        )
        self.pathfinder.computeShortestPath()

    def run(self):
        while not self.should_shutdown:
            self._setup_pathfinder()
            self.is_running = True
            while self.is_running:
                self.update_step()
        self.shutdown()

    def update_step(self):
        # 1) Choose next move
        pos_r, pos_c = self.simulator.guided_miner_pos
        path = self.pathfinder.getShortestPath() or []
        if len(path) > 1:
            nx, ny = path[1]
            move   = (ny - pos_r, nx - pos_c)
            act    = MOVE_TO_ACTION_MAP.get(move)
        elif self.simulator.guided_miner_pos in self.simulator.goal_positions:
            print("[SUCCESS] Reached goal.")
            self.is_running = False
            return
        else:
            act = None

        # 2) Step simulator
        new_state = self.simulator.step(guided_miner_action=act)
        self.path_history.append(new_state['guided_miner_pos'])

        # 3) Update planner's start
        old_pos = (pos_c, pos_r)
        nr, nc = new_state['guided_miner_pos']
        new_pos = (nc, nr)
        if new_pos != old_pos:
            self.pathfinder.updateStart(new_pos[0], new_pos[1])

        # 4) Detect sensors crossing a tier boundary
        dirty_cells = []
        for s_pos, batt in self.simulator.sensor_batteries.items():
            old_t = self.sensor_previous_costs[s_pos]
            new_t = sensor_cost_tier(batt)
            if new_t != old_t:
                dirty_cells.extend(self.simulator.sensor_cell_map[s_pos])
                self.sensor_previous_costs[s_pos] = new_t

        # 5) Repair those vertices
        for (x, y) in dirty_cells:
            sensor = self.simulator.cell_to_sensor[(x,y)]
            new_cost = sensor_cost_tier(self.simulator.sensor_batteries[sensor])
            self.cost_map[y, x] = new_cost 
            self.pathfinder.updateVertex(x, y)
            for (nx, ny) in self.pathfinder.neighbors(x, y):
                self.pathfinder.updateVertex(nx, ny)

        # 6) Replan
        self.pathfinder.computeShortestPath()

        # 7) Render if needed
        if self.simulator.render_mode == 'human':
            ok = self.simulator.render(
                show_miners=False,
                dstar_path=self.pathfinder.getShortestPath(),
                path_history=self.path_history
            )
            if not ok:
                print("[INFO] Window closed by user.")
                self.is_running       = False
                self.should_shutdown = True

    def shutdown(self):
        print("--- Shutting down simulations ---")
        self.simulator.close()
