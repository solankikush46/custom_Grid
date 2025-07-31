# SimulationController.py

import os

from .MineSimulator import MineSimulator
from .DStarLite import DStarLite
from .constants import MOVE_TO_ACTION_MAP

def sensor_cost_tier(batt):
    if batt <= 5:    return 400.0
    if batt <= 10:   return 200.0
    if batt <= 30:   return 50.0
    return 0.0

class SimulationController:
    """
    Integrates MineSimulator with D* Lite.
    Chooses the closest/safest of multiple goals,
    and only repairs the planner on cells whose sensor-tier changed.
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
        # 3) build static obstacle set
        static_obs = { (c, r) for (r, c) in self.simulator.impassable_positions }
        W, H     = self.simulator.n_cols, self.simulator.n_rows
        # 4) start in (x,y)
        sr, sc   = state['guided_miner_pos']
        start    = (sc, sr)
        # 5) ALL goals as list of (x,y)
        goals    = [ (c, r) for (r, c) in state['goal_positions'] ]

        # 6) init D* Lite with multi-goal support
        self.pathfinder = DStarLite(
            width=W,
            height=H,
            start=start,
            goal=goals,
            cost_function=self._battery_cost,
            known_obstacles=static_obs,
            heuristic=None
        )
        self.pathfinder.compute_shortest_path()

    def _battery_cost(self, cell_xy):
        """
        O(1) lookup: cell_xy→owner sensor→battery→cost tier
        """
        sensor = self.simulator.cell_to_sensor[cell_xy]
        batt   = self.simulator.sensor_batteries[sensor]
        return sensor_cost_tier(batt)

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
        path = self.pathfinder.get_shortest_path() or []
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

        # 3) Update D* Lite’s start & km
        old = self.pathfinder.start
        nr, nc = new_state['guided_miner_pos']
        new = (nc, nr)
        if new != old:
            self.pathfinder.start = new
            self.pathfinder.km   += self.pathfinder.h(old, new)

        # 4) Detect which sensors crossed a tier boundary
        dirty_cells = []
        for s_pos, batt in self.simulator.sensor_batteries.items():
            old_tier = self.sensor_previous_costs[s_pos]
            new_tier = sensor_cost_tier(batt)
            if new_tier != old_tier:
                # mark all cells that belong to this sensor
                dirty_cells.extend(self.simulator.sensor_cell_map[s_pos])
                self.sensor_previous_costs[s_pos] = new_tier

        # 5) Repair only those vertices
        for cell in dirty_cells:
            self.pathfinder.update_vertex(cell)
            for p in self.pathfinder.pred(cell):
                self.pathfinder.update_vertex(p)

        # 6) Replan
        self.pathfinder.compute_shortest_path()

        # 7) Render if needed
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
