# SimulationController.py

import os
import numpy as np
from sb3_contrib import RecurrentPPO

from .MineSimulator import MineSimulator
from .DStarLite.DStarLite import DStarLite  # C++ pybind11 module
from .constants import MOVE_TO_ACTION_MAP

##==============================================================
## Helpers
##==============================================================
def sensor_cost_tier(batt):
    if batt <= 5:    return 400.0
    if batt <= 10:   return 200.0
    if batt <= 30:   return 50.0
    return 0.0

##==============================================================
## SimulationController Class
##==============================================================
# SimulationController.py

import os
import numpy as np
from sb3_contrib import RecurrentPPO

from .MineSimulator import MineSimulator
from .DStarLite.DStarLite import DStarLite  # C++ pybind11 module
from .constants import MOVE_TO_ACTION_MAP

def sensor_cost_tier(batt: float) -> float:
    """
    Maps a raw battery level (0–100) to a movement penalty cost.
    """
    if batt <= 5:    return 400.0
    if batt <= 10:   return 200.0
    if batt <= 30:   return 50.0
    return 0.0

class SimulationController:
    """
    Integrates MineSimulator with the C++ D* Lite planner and an optional SB3 battery predictor.
    Implements a time-dependent cost map: for each cell, costs are based on either
    (a) the predictor's forecast at the Chebyshev-distance arrival time, or
    (b) a constant drain_rate * distance if `predicted_depletion_rate` is provided.
    """
    def __init__(
        self,
        experiment_folder: str,
        render: bool = True,
        show_predicted: bool = True,
        predicted_depletion_rate: float = None
    ):
        """
        Args:
            experiment_folder: e.g. "mine_50x50_20miners"
            render: if True, create a Pygame window
            show_predicted: if True, overlay predicted battery percentages
            predicted_depletion_rate: if not None, use batt_pred = batt_current - predicted_depletion_rate * distance
        """
        parts       = experiment_folder.split('_')
        grid_file   = f"{parts[0]}_{parts[1]}.txt"
        n_miners    = int(parts[-1].replace('miners',''))
        render_mode = 'human' if render else None

        # 1) Core simulator
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode
        )

        # Sensor → index for quick lookup
        self.sensor_index = {
            pos: i for i, pos in enumerate(self.simulator.sensor_positions)
        }
        self.num_sensors = len(self.simulator.sensor_positions)

        # 2) Load the latest RecurrentPPO battery predictor
        base = os.path.join("saved_experiments", experiment_folder)
        try:
            runs = sorted(
                d for d in os.listdir(base) if d.startswith("RecurrentPPO_")
            )
        except FileNotFoundError:
            raise RuntimeError(f"No such experiment folder: {base}")
        if not runs:
            raise RuntimeError(f"No RecurrentPPO runs in {base}")
        model_dir  = runs[-1]
        model_path = os.path.join(base, model_dir, "model.zip")
        self.predictor = RecurrentPPO.load(model_path)

        # Internal state
        self.pathfinder            = None
        self.cost_map              = None
        self.path_history          = []
        self.is_running            = False
        self.should_shutdown       = False
        self.sensor_previous_costs = {}
        self.last_pred_norm        = None
        self.current_start         = None  # (x, y)
        self.show_predicted        = show_predicted
        self.predicted_depletion_rate        = predicted_depletion_rate

    def _setup_pathfinder(self):
        state = self.simulator.reset()
        self.path_history = [state['guided_miner_pos']]

        # Perfect first prediction (error = 0)
        init_batts = [state['sensor_batteries'][pos]
                      for pos in self.simulator.sensor_positions]
        self.last_pred_norm = np.array(init_batts, dtype=np.float32) / 100.0

        # Record initial tiers
        for pos, batt in state['sensor_batteries'].items():
            self.sensor_previous_costs[pos] = sensor_cost_tier(batt)

        # Initialize cost map (H x W)
        H, W = self.simulator.n_rows, self.simulator.n_cols
        self.cost_map = np.zeros((H, W), dtype=np.float64)

        # Static obstacles
        static_obs = [(x, y) for (y, x) in self.simulator.impassable_positions]

        # Start & goals
        r0, c0 = state['guided_miner_pos']
        start_x, start_y = c0, r0
        self.current_start = (start_x, start_y)
        goals = [(c, r) for (r, c) in state['goal_positions']]

        # Create D* Lite planner
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
        # 1) Choose next move from D* Lite
        r0, c0 = self.simulator.guided_miner_pos
        path = self.pathfinder.getShortestPath() or []
        if len(path) > 1:
            nx, ny = path[1]
            move   = (ny - r0, nx - c0)
            act    = MOVE_TO_ACTION_MAP.get(move)
        elif (r0, c0) in self.simulator.goal_positions:
            self.is_running = False
            return
        else:
            act = None

        # 2) Step simulator
        new_state = self.simulator.step(guided_miner_action=act)
        self.path_history.append(new_state['guided_miner_pos'])

        # 3) Build RL predictor observation
        batts_norm = np.array([
            new_state['sensor_batteries'][pos]
            for pos in self.simulator.sensor_positions
        ], dtype=np.float32) / 100.0

        # Connection counts
        conns = {pos: 0 for pos in self.simulator.sensor_positions}
        movers = new_state['miner_positions'] + [new_state['guided_miner_pos']]
        for mv in movers:
            sensor = self.simulator.cell_to_sensor[mv]
            conns[sensor] += 1
        miner_norm = np.array(
            [conns[pos] for pos in self.simulator.sensor_positions],
            dtype=np.float32
        ) / self.simulator.n_miners

        err_norm = batts_norm - self.last_pred_norm
        obs = np.stack([batts_norm, miner_norm, err_norm], axis=1).flatten()

        # 4) Compute Chebyshev distances from start
        H, W = self.simulator.n_rows, self.simulator.n_cols
        sx, sy = self.current_start
        ys = np.arange(H)[:, None]
        xs = np.arange(W)[None, :]
        D  = np.maximum(np.abs(xs - sx), np.abs(ys - sy)).astype(int)  # shape (H,W)
        maxD = D.max()

        # 5) Forecast or constant‐drain
        S = self.num_sensors
        batt_map = np.zeros((H, W), dtype=np.float32)
        if self.predicted_depletion_rate is None:
            # learned predictor rollout
            preds = np.zeros((S, maxD + 1), dtype=np.float32)
            preds[:, 0] = batts_norm
            last = preds[:, 0]
            for t in range(1, maxD + 1):
                obs_t = np.stack([last, miner_norm, np.zeros_like(last)], axis=1).flatten()
                next_norm, _ = self.predictor.predict(obs_t, deterministic=True)
                preds[:, t] = next_norm
                last = next_norm
            self.last_pred_norm = batts_norm
            # Build batt_map from preds
            for y in range(H):
                for x in range(W):
                    sensor = self.simulator.cell_to_sensor[(x, y)]
                    idx = self.sensor_index[sensor]
                    batt_map[y, x] = float(preds[idx, D[y, x]] * 100.0)
        else:
            # simple constant drain: batt_pred = max(current - rate * dist, 0)
            for y in range(H):
                for x in range(W):
                    sensor = self.simulator.cell_to_sensor[(x, y)]
                    idx = self.sensor_index[sensor]
                    batt_current = batts_norm[idx] * 100.0
                    batt_pred = max(batt_current - self.predicted_depletion_rate * D[y, x], 0.0)
                    batt_map[y, x] = batt_pred
            # last_pred_norm not used in this mode

        # 6) Update cost_map & collect dirty cells
        dirty = []
        for y in range(H):
            for x in range(W):
                tier = sensor_cost_tier(batt_map[y, x])
                if self.cost_map[y, x] != tier:
                    self.cost_map[y, x] = tier
                    dirty.append((x, y))

        # 7) Notify D* Lite of cost changes
        for x, y in dirty:
            self.pathfinder.updateVertex(x, y)
            for nx, ny in self.pathfinder.neighbors(x, y):
                self.pathfinder.updateVertex(nx, ny)

        # 8) If miner moved, update start
        r1, c1 = new_state['guided_miner_pos']
        if (c1, r1) != (c0, r0):
            self.pathfinder.updateStart(c1, r1)
            self.current_start = (c1, r1)

        # 9) Replan
        self.pathfinder.computeShortestPath()

        # 10) Render
        if self.simulator.render_mode == 'human':
            ok = self.simulator.render(
                show_miners           = False,
                dstar_path            = self.pathfinder.getShortestPath(),
                path_history          = self.path_history,
                predicted_battery_map = batt_map if self.show_predicted else None
            )
            if not ok:
                print("[INFO] Window closed by user.")
                self.is_running       = False
                self.should_shutdown = True

    def shutdown(self):
        print("--- Shutting down simulations ---")
        self.simulator.close()

##==============================================================
##
##==============================================================
def estimate_sensor_depletion_rate(experiment_folder: str,
                                   n_episodes: int = 100):
    """
    Run `n_episodes` headless simulations (reset→goal) under the given
    `experiment_folder`, track each sensor's battery drop per timestep,
    and return:
       mean_rate : float   average % depletion per step (across sensors & episodes)
       std_rate  : float   sample‐stddev of the per‐episode averages
       rates      : np.ndarray  length‐n_episodes list of per‐episode avg rates
    """
    rates = []

    for ep in range(1, n_episodes + 1):
        ctrl = SimulationController(experiment_folder, render=False)
        ctrl._setup_pathfinder()
        sim = ctrl.simulator

        # initialize per-sensor history
        history = { pos: [b] for pos, b in sim.sensor_batteries.items() }

        # run until goal reached
        while True:
            ctrl.update_step()
            for pos in history:
                history[pos].append(sim.sensor_batteries[pos])
            if not ctrl.is_running:
                break

        # total steps in this episode
        T = len(next(iter(history.values()))) - 1
        if T <= 0:
            continue

        # compute per‐sensor depletion rate = (start−end)/T
        sensor_rates = [
            (vals[0] - vals[-1]) / T
            for vals in history.values()
        ]
        # average over sensors
        ep_rate = np.mean(sensor_rates)
        rates.append(ep_rate)
        print(f"[Episode {ep}/{n_episodes}] avg depletion = {ep_rate:.3f}%/step")

    rates = np.array(rates, dtype=float)
    mean_rate = float(rates.mean())
    std_rate  = float(rates.std(ddof=1)) if len(rates) > 1 else 0.0
    return mean_rate, std_rate, rates

def report_depletion_rate(experiment_folder: str, n_episodes: int = 100):
    """
    Runs `n_episodes` and prints a tidy summary of the mean ± std depletion rate.
    Returns the tuple (mean_rate, std_rate, all_rates) if you need to inspect it.
    """
    mean_rate, std_rate, all_rates = estimate_sensor_depletion_rate(
        experiment_folder, n_episodes=n_episodes
    )
    print(f"\n=== RESULTS over {n_episodes} episodes on '{experiment_folder}' ===")
    print(f"Mean depletion rate: {mean_rate:.3f}% per step")
    print(f"Std  deviation     : {std_rate:.3f}% per step")
    return mean_rate, std_rate, all_rates

