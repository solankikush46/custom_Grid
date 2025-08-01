# SimulationController.py

import os
import numpy as np
import json
from sb3_contrib import RecurrentPPO
from collections import defaultdict

from .MineSimulator import MineSimulator
from .DStarLite.DStarLite import DStarLite
from .constants import *

##==============================================================
## Helpers
##==============================================================
def sensor_cost_tier(batt: float) -> float:
    """
    Maps a raw battery level (0–100) to a movement penalty cost.
    """
    if batt <= 5:
        return 400.0
    if batt <= 10:
        return 200.0
    if batt <= 20:
        return 100.0
    if batt <= 30:
        return 50.0
    return 0.0

##==============================================================
## SimulationController Class
##==============================================================
class SimulationController:
    """
    Integrates MineSimulator with D* Lite planner and battery forecasting.

    Modes:
      - 'static': use current battery only.
      - 'constant_rate': load per-sensor average depletion rates from JSON.
      - 'model': use recurrent model to forecast battery.

    If get_average_depletion=True, runs num_episodes to collect deltas,
    then saves avg_sensor_depletion.json & battery_deltas.json.
    """

    def __init__(
        self,
        experiment_folder: str,
        render: bool = True,
        show_miners: bool = False,
        show_predicted: bool = True,
        mode: str = "static",
        get_average_depletion: bool = False,
        num_episodes: int = 1
    ):
        # Validate mode
        if mode not in ("static", "constant_rate", "model"):
            raise ValueError(f"Unsupported mode '{mode}'")

        if get_average_depletion and num_episodes < 1:
            raise ValueError(
                "num_episodes must be >= 1 when collecting averages"
            )

        # Settings
        self.experiment_folder     = experiment_folder
        self.show_predicted        = show_predicted
        self.show_miners           = show_miners
        self.mode                  = mode
        self.get_average_depletion = get_average_depletion
        self.num_episodes          = num_episodes

        # Parse grid file & miners
        parts       = experiment_folder.split('_')
        grid_file   = f"{parts[0]}_{parts[1]}.txt"
        n_miners    = int(parts[-1].replace('miners',''))
        render_mode = 'human' if render else None

        # Core simulator
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode,
            show_predicted=show_predicted
        )

        # Sensor indexing
        self.sensor_index = {
            pos: i
            for i, pos in enumerate(self.simulator.sensor_positions)
        }
        self.num_sensors = len(self.simulator.sensor_positions)

        # If constant_rate, load rates from JSON
        self.constant_rates = {}
        if self.mode == 'constant_rate':
            path = os.path.join(
                SAVE_DIR,
                experiment_folder,
                'avg_sensor_depletion.json'
            )
            if not os.path.isfile(path):
                raise RuntimeError(f"Average depletion file not found at {path}")

            with open(path, 'r') as f:
                data = json.load(f)

            # parse keys "r,c" -> (r,c)
            for k, v in data.items():
                r, c = map(int, k.split(','))
                self.constant_rates[(r, c)] = float(v)

        # Predictor for 'model'
        self.predictor = None
        if self.mode == 'model':
            base = os.path.join(SAVE_DIR, experiment_folder)
            runs = sorted(
                d for d in os.listdir(base)
                if d.startswith('RecurrentPPO_')
            )
            if not runs:
                raise RuntimeError(f"No model runs in {base}")

            model_path = os.path.join(
                base,
                runs[-1],
                'model.zip'
            )
            self.predictor = RecurrentPPO.load(model_path)

        # Planning state
        self.pathfinder            = None
        self.cost_map              = None
        self.path_history          = []
        self.is_running            = False
        self.should_shutdown       = False
        self.sensor_previous_costs = {}
        self.last_pred_norm        = None
        self.current_start         = None

        # Data collection
        self.battery_deltas = {
            pos: []
            for pos in self.simulator.sensor_positions
        }


    def _setup_pathfinder(self):
        # Reset simulator
        state = self.simulator.reset()
        self.path_history = [state['guided_miner_pos']]

        # Initialize normalized battery vector
        init_batts = [
            state['sensor_batteries'][pos]
            for pos in self.simulator.sensor_positions
        ]
        self.last_pred_norm = (
            np.array(init_batts, dtype=np.float32) / 100.0
        )

        # Record previous cost tiers
        for pos, batt in state['sensor_batteries'].items():
            self.sensor_previous_costs[pos] = sensor_cost_tier(batt)

        # Build cost map
        H, W = self.simulator.n_rows, self.simulator.n_cols
        self.cost_map = np.zeros((H, W), dtype=np.float64)
        static_obs = [
            (x, y)
            for (y, x) in self.simulator.impassable_positions
        ]

        # Setup planner
        r0, c0 = state['guided_miner_pos']
        self.current_start = (c0, r0)
        goals = [
            (c, r)
            for (r, c) in state['goal_positions']
        ]

        self.pathfinder = DStarLite(
            W,
            H,
            c0,
            r0,
            goals,
            self.cost_map,
            static_obs
        )
        self.pathfinder.computeShortestPath()


    def run(self):
        eps = 0
        while not self.should_shutdown:
            eps += 1
            self._setup_pathfinder()
            self.is_running = True

            while self.is_running:
                self.update_step()

            if (
                self.get_average_depletion and
                eps >= self.num_episodes
            ):
                break

        self.shutdown()


    def update_step(self):
        # Choose action
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

        # Step simulator
        new_state = self.simulator.step(
            guided_miner_action=act
        )
        self.path_history.append(
            new_state['guided_miner_pos']
        )

        # Collect deltas
        if (
            self.get_average_depletion and
            'battery_deltas' in new_state
        ):
            for pos, delta in new_state['battery_deltas'].items():
                self.battery_deltas[pos].append(float(delta))

        # Build features
        batts_norm = (
            np.array([
                new_state['sensor_batteries'][pos]
                for pos in self.simulator.sensor_positions
            ], dtype=np.float32) / 100.0
        )

        conns = {
            pos: 0
            for pos in self.simulator.sensor_positions
        }
        for mv in (
            new_state['miner_positions'] +
            [new_state['guided_miner_pos']]
        ):
            conns[self.simulator.cell_to_sensor[mv]] += 1

        miner_norm = (
            np.array([
                conns[pos]
                for pos in self.simulator.sensor_positions
            ], dtype=np.float32) / self.simulator.n_miners
        )

        err_norm = batts_norm - self.last_pred_norm
        obs      = np.stack([
            batts_norm,
            miner_norm,
            err_norm
        ], axis=1).flatten()

        # Compute distance map
        H, W = self.simulator.n_rows, self.simulator.n_cols
        ys    = np.arange(H)[:, None]
        xs    = np.arange(W)[None, :]
        D     = np.maximum(
            np.abs(xs - self.current_start[0]),
            np.abs(ys - self.current_start[1])
        ).astype(int)
        maxD  = D.max()

        # Build battery map
        batt_map = np.zeros((H, W), dtype=np.float32)
        if self.mode == 'model':
            S     = self.num_sensors
            preds = np.zeros((S, maxD + 1), dtype=np.float32)
            preds[:, 0] = batts_norm
            last        = preds[:, 0]

            for t in range(1, maxD + 1):
                obs_t     = np.stack([
                    last,
                    miner_norm,
                    np.zeros_like(last)
                ], axis=1).flatten()
                next_norm, _ = self.predictor.predict(
                    obs_t, deterministic=True
                )
                preds[:, t] = next_norm
                last        = next_norm

            self.last_pred_norm = batts_norm

            for y, x in self.simulator.free_cells:
                idx            = self.sensor_index[
                    self.simulator.cell_to_sensor[(x, y)]
                ]
                batt_map[y, x] = float(preds[idx, D[y, x]] * 100.0)

        elif self.mode == 'constant_rate':
            rates = self.constant_rates
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                rate   = rates.get(sensor, 0.0)
                curr   = batts_norm[
                    self.sensor_index[sensor]
                ] * 100.0
                batt_map[y, x] = max(curr - rate * D[y, x], 0.0)

        else:
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                batt_map[y, x] = float(
                    batts_norm[self.sensor_index[sensor]] * 100.0
                )

        # Update cost map
        dirty = []
        for y, x in self.simulator.free_cells:
            tier = sensor_cost_tier(batt_map[y, x])
            if self.cost_map[y, x] != tier:
                self.cost_map[y, x] = tier
                dirty.append((x, y))

        for x, y in dirty:
            self.pathfinder.updateVertex(x, y)
            for nx, ny in self.pathfinder.neighbors(x, y):
                self.pathfinder.updateVertex(nx, ny)

        # Update start if moved
        r1, c1 = new_state['guided_miner_pos']
        if (c1, r1) != (c0, r0):
            self.pathfinder.updateStart(c1, r1)
            self.current_start = (c1, r1)

        # Replan
        self.pathfinder.computeShortestPath()

        # Render
        if self.simulator.render_mode == 'human':
            ok = self.simulator.render(
                show_miners           = self.show_miners,
                dstar_path            = self.pathfinder.getShortestPath(),
                path_history          = self.path_history,
                predicted_battery_map = batt_map if self.show_predicted else None
            )

            if not ok:
                print("[INFO] Window closed by user.")
                self.is_running       = False
                self.should_shutdown = True


    def compute_average_depletion(self):
        return {
            pos: (sum(vals)/len(vals) if vals else 0.0)
            for pos, vals in self.battery_deltas.items()
        }


    def get_sensor_delta_series(self):
        return self.battery_deltas


    def shutdown(self):
        if self.get_average_depletion:
            try:
                avg = self.compute_average_depletion()
                base = os.path.join(
                    SAVE_DIR,
                    self.experiment_folder
                )
                os.makedirs(base, exist_ok=True)
                avg_path = os.path.join(
                    base,
                    'avg_sensor_depletion.json'
                )
                with open(avg_path, 'w') as f:
                    json.dump(
                        {f"{p[0]},{p[1]}": v for p, v in avg.items()},
                        f,
                        indent=2
                    )
                print(f"[INFO] Saved average depletion to {avg_path}")

            except Exception as e:
                print(f"[ERROR] Saving average depletion failed: {e}")

            try:
                deltas_path = os.path.join(
                    SAVE_DIR,
                    self.experiment_folder,
                    'battery_deltas.json'
                )
                with open(deltas_path, 'w') as f:
                    json.dump(
                        {f"{p[0]},{p[1]}": vals for p, vals in self.battery_deltas.items()},
                        f,
                        indent=2
                    )
                print(f"[INFO] Saved raw battery deltas to {deltas_path}")

            except Exception as e:
                print(f"[ERROR] Saving raw battery deltas failed: {e}")

        print("--- Shutting down simulations ---")
        if self.simulator:
            self.simulator.close()
