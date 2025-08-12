# MineEnv.py

import os
import json

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .MineSimulator import MineSimulator
from .DStarLite.DStarLite import DStarLite
from .constants import SAVE_DIR, MOVE_TO_ACTION_MAP, COST_TABLE
from .utils import parse_experiment_data
# from .reward_functions import compute_reward  # must return (reward: float, subrewards: dict)

#===============================
# Helpers
#===============================
def cheb(a, b):
    """Chebyshev distance between two (r, c) cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def sensor_cost_tier(batt):
    """
    Map raw battery (0–100) to movement penalty via COST_TABLE with clamping.
    """
    return COST_TABLE[int(batt)]


def _sorted_allowed_moves():
    """
    Build the discrete action set directly from MOVE_TO_ACTION_MAP keys
    (keeps you in sync with constants.DIRECTION_MAP). Filters None-mapped moves.
    """
    items = [(drc, tok) for drc, tok in MOVE_TO_ACTION_MAP.items() if tok is not None]
    items.sort(key=lambda it: (it[0][0], it[0][1]))  # stable order
    return [drc for drc, _ in items]

##==============================================================
## MineEnv Class
##==============================================================
class MineEnv(gym.Env):
    """
    Gym environment integrating MineSimulator with a D* Lite cost map derived from sensor batteries.

    Modes:
      - 'static'        : use current sensor batteries as-is
      - 'constant_rate' : decrease per-sensor battery at Chebyshev ETA using
                          rates from avg_sensor_depletion.json (in SAVE_DIR/<exp>/)

    Reward is delegated to reward_functions.compute_reward(env) -> (reward, subrewards).

    Manual depletion collection:
      - Call depletion_reset() before a run you want to measure.
      - On each step, call depletion_update_from_info(info) or depletion_update(...).
      - Call depletion_save() to write JSONs under SAVE_DIR/<experiment>/.

    Notes:
      - No step-based truncation; episodes end only when the guided miner reaches any goal.
      - D* is optional and used only for on-screen overlay; RL chooses actions.
    """

    metadata = {"render.modes": ["human", "none"]}

    def __init__(
        self,
        experiment_folder,
        render=False,
        show_miners=False,
        show_predicted=True,
        mode="static",              # 'static' | 'constant_rate'
        # The following three are kept as public knobs for your reward fn to read:
        step_penalty=-0.01,
        goal_bonus=10.0,
        collision_penalty=-1.0,
        use_planner_overlay=False   # keep D* path for rendering only
    ):
        super().__init__()

        if mode not in ("static", "constant_rate"):
            raise ValueError("mode must be 'static' or 'constant_rate'")

        # ---- Parse experiment folder via utils ----
        data = parse_experiment_data(experiment_folder)
        exp   = data["experiment"]
        grid  = data["grid"]
        paths = data["paths"]
        arts  = data["artifacts"]

        grid_file = grid["file"]            # e.g., "mine_50x50.txt"
        n_miners  = exp["n_miners"]
        self.norm = float(exp["norm"])      # max(rows, cols) – used for goal normalization

        # Config
        self.experiment_folder = experiment_folder
        self.render_enabled = bool(render)
        self.show_miners = bool(show_miners)
        self.show_predicted = bool(show_predicted)
        self.mode = mode
        # expose penalties/bonuses for reward_functions to read
        self.step_penalty = float(step_penalty)
        self.goal_bonus = float(goal_bonus)
        self.collision_penalty = float(collision_penalty)
        self.use_planner_overlay = bool(use_planner_overlay)

        # --- manual depletion collection state ---
        self._delta_sums = {}         # {(r,c): sum of drops}
        self._delta_counts = {}       # {(r,c): count of contributing steps}
        self._prev_sensor_batts = {}  # {(r,c): last seen value}

        render_mode = 'human' if self.render_enabled else None

        # Simulator
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode,
            show_predicted=self.show_predicted
        )

        # Stable sensor ordering
        self.sensor_index = {pos: i for i, pos in enumerate(self.simulator.sensor_positions)}

        # Constant-rate data (prefer parsed map from utils; fallback to file)
        self.constant_rates = {}
        if self.mode == "constant_rate":
            if arts.get("avg_depletion_map"):
                self.constant_rates = dict(arts["avg_depletion_map"])  # {(r,c): rate}
            else:
                path = paths["avg_depletion_json"]
                if not os.path.isfile(path):
                    raise RuntimeError(f"Average depletion file not found at {path}")
                with open(path, "r") as f:
                    data_json = json.load(f)
                for k, v in data_json.items():
                    r, c = map(int, k.split(","))
                    self.constant_rates[(r, c)] = float(v)

        # Planner for overlay / cost plumbing
        self.pathfinder = None
        self.cost_map = None
        self.current_start_xy = None  # (c, r)
        self.path_history = []

        # Episode tracking (no hard truncation)
        self._steps = 0
        self._cumulative_reward = 0.0
        self._visited = set()
        self._revisit_count = 0
        self._obstacle_hits = 0
        self._prev_goal_dist = 0

        # Per-step signals (reward fn can read these)
        self.last_distance_to_goal = 0.0
        self.last_progress = 0.0
        self.last_obstacle_hit = False
        self.last_reached_goal = False

        # Action space derived from your mapping
        self._ACTIONS = _sorted_allowed_moves()
        if not self._ACTIONS:
            self._ACTIONS = [(0, 0)]
        self.action_space = spaces.Discrete(len(self._ACTIONS))

        # Observation space — compact vector:
        # vec = [dx_goal_norm, dy_goal_norm, d_goal_norm, batt_min, batt_mean, batt_max]
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(
                low=np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([ 1,  1, 1, 100, 100, 100], dtype=np.float32),
                dtype=np.float32
            )
        })

    # ========================= Gym API =========================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Episode stats
        self._steps = 0
        self._cumulative_reward = 0.0
        self._visited.clear()
        self._revisit_count = 0
        self._obstacle_hits = 0

        # Reset sim
        state = self.simulator.reset()
        agent_rc = tuple(state["guided_miner_pos"])
        self.path_history = [agent_rc]

        # Recompute normalization in case dims changed
        self.norm = float(max(self.simulator.n_rows, self.simulator.n_cols))

        # Planner+cost map (optional overlay)
        H, W = self.simulator.n_rows, self.simulator.n_cols
        self.cost_map = np.zeros((H, W), dtype=np.float64)
        static_obs = [(x, y) for (y, x) in self.simulator.impassable_positions]
        r0, c0 = agent_rc
        self.current_start_xy = (c0, r0)
        goals_xy = [(c, r) for (r, c) in self.simulator.goal_positions]

        if self.use_planner_overlay:
            self.pathfinder = DStarLite(W, H, c0, r0, goals_xy, self.cost_map, static_obs)
            self.pathfinder.computeShortestPath()
        else:
            self.pathfinder = None

        # First battery map + cost updates
        batt_map = self._build_batt_map_and_update_costs(state)
        if self.use_planner_overlay and self.pathfinder is not None:
            self._replan_from(agent_rc)

        # Initial obs
        self._prev_goal_dist = self._closest_goal_dist(agent_rc)
        self.last_distance_to_goal = float(self._prev_goal_dist)
        self.last_progress = 0.0
        self.last_obstacle_hit = False
        self.last_reached_goal = False

        # Info with initial battery on current cell
        curr_batt = self._current_cell_battery(state, agent_rc)
        obs = self._make_obs(agent_rc, state.get("sensor_batteries", {}))
        info = {
            "sensor_batteries": state.get("sensor_batteries", {}),
            "distance_to_goal": float(self._prev_goal_dist),
            "current_battery": float(curr_batt) if curr_batt is not None else None,
            "terminated": False,
            "truncated": False,
        }

        if self.render_enabled and self.simulator.render_mode == "human":
            self._render_frame(batt_map)

        return obs, info

    def step(self, action):
        if self.cost_map is None or self.current_start_xy is None:
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        self._steps += 1

        # Decode action → simulator token
        try:
            dr, dc = self._ACTIONS[int(action)]
        except Exception:
            dr, dc = (0, 0)
        sim_token = MOVE_TO_ACTION_MAP.get((dr, dc), None)

        # Step simulator
        try:
            s = self.simulator.step(guided_miner_action=sim_token)
        except Exception as e:
            agent_rc = tuple(self.simulator.guided_miner_pos)
            obs = self._make_obs(agent_rc, {})
            info = {"error": str(e), "terminated": True, "truncated": False}
            return obs, -5.0, True, False, info

        agent_rc = tuple(s["guided_miner_pos"])
        self.path_history.append(agent_rc)

        # Visit stats
        if agent_rc in self._visited:
            self._revisit_count += 1
        else:
            self._visited.add(agent_rc)

        if s.get("obstacle_hit", False):
            self._obstacle_hits += 1

        # Battery->cost map + (optional) replan for overlay only
        self.current_start_xy = (agent_rc[1], agent_rc[0])  # (c, r)
        batt_map = self._build_batt_map_and_update_costs(s)
        if self.use_planner_overlay and self.pathfinder is not None:
            self._replan_from(agent_rc)

        # --------- signals for reward function ---------
        d_now = self._closest_goal_dist(agent_rc)
        self.last_distance_to_goal = float(d_now)
        self.last_progress = float(self._prev_goal_dist - d_now)
        self.last_obstacle_hit = bool(s.get("obstacle_hit", False))
        self.last_reached_goal = bool(agent_rc in self.simulator.goal_positions)

        # --------- delegate reward to reward_functions ---------
        try:
            # reward, sub = compute_reward(self)
            reward, sub = 0.0, {}
        except Exception as e:
            # Safe fallback: no crash during training
            reward = 0.0
            sub = {"reward_fn_error": str(e)}

        # Episode status
        terminated = self.last_reached_goal
        truncated = False  # no hard cap

        self._cumulative_reward += float(reward)
        self._prev_goal_dist = d_now

        # Render
        if self.render_enabled and self.simulator.render_mode == "human":
            self._render_frame(batt_map)

        # Obs + info (includes current_battery for TB)
        curr_batt = self._current_cell_battery(s, agent_rc)
        obs = self._make_obs(agent_rc, s.get("sensor_batteries", {}))
        info = {
            "sensor_batteries": s.get("sensor_batteries", {}),
            "current_reward": float(reward),
            "distance_to_goal": float(d_now),
            "current_battery": float(curr_batt) if curr_batt is not None else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "subrewards": sub,
        }
        if terminated or truncated:
            avg_batt = 0.0
            sbs = s.get("sensor_batteries", {})
            if sbs:
                vals = np.array(list(sbs.values()), dtype=np.float32)
                avg_batt = float(vals.mean())
            info.update({
                "cumulative_reward": float(self._cumulative_reward),
                "obstacle_hits": int(self._obstacle_hits),
                "visited_count": int(len(self._visited)),
                "average_battery_level": float(avg_batt),
                "episode_length": int(self._steps),
                "revisit_count": int(self._revisit_count),
            })

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        # Rendering is driven inside step() for perf; keep Gym signature.
        return None

    def close(self):
        try:
            self.simulator.close()
        except Exception:
            pass

    # ========================= Depletion collection (manual) =========================
    def depletion_reset(self):
        self._delta_sums = {}
        self._delta_counts = {}
        self._prev_sensor_batts = {}

    def depletion_update(self, sensor_batteries, battery_deltas=None, clamp_negatives=True):
        if battery_deltas and isinstance(battery_deltas, dict):
            for pos, d in battery_deltas.items():
                try:
                    dv = float(d)
                except Exception:
                    dv = 0.0
                if clamp_negatives and dv < 0.0:
                    dv = 0.0
                self._delta_sums[pos] = self._delta_sums.get(pos, 0.0) + dv
                self._delta_counts[pos] = self._delta_counts.get(pos, 0) + 1
        else:
            cur = dict(sensor_batteries or {})
            for pos, v_now in cur.items():
                v_prev = self._prev_sensor_batts.get(pos, v_now)
                dv = float(v_prev - v_now)
                if clamp_negatives and dv < 0.0:
                    dv = 0.0
                self._delta_sums[pos] = self._delta_sums.get(pos, 0.0) + dv
                self._delta_counts[pos] = self._delta_counts.get(pos, 0) + 1
            self._prev_sensor_batts = cur

    def depletion_update_from_info(self, info, clamp_negatives=True):
        sensor_batts = info.get("sensor_batteries", {})
        battery_deltas = info.get("battery_deltas", None)
        self.depletion_update(sensor_batts, battery_deltas, clamp_negatives=clamp_negatives)

    def depletion_save(self, out_dir=None):
        if out_dir is None:
            out_dir = os.path.join(SAVE_DIR, self.experiment_folder)
        os.makedirs(out_dir, exist_ok=True)

        # Averages
        avgs = {}
        for pos in self.simulator.sensor_positions:
            s = self._delta_sums.get(pos, 0.0)
            n = self._delta_counts.get(pos, 0)
            avgs[pos] = (s / n) if n > 0 else 0.0

        avg_path = os.path.join(out_dir, 'avg_sensor_depletion.json')
        with open(avg_path, 'w') as f:
            json.dump({f"{p[0]},{p[1]}": v for p, v in avgs.items()}, f, indent=2)

        raw_path = os.path.join(out_dir, 'battery_deltas.json')
        with open(raw_path, 'w') as f:
            json.dump({f"{p[0]},{p[1]}": [self._delta_sums.get(p, 0.0),
                                          self._delta_counts.get(p, 0)]
                       for p in self.simulator.sensor_positions}, f, indent=2)

        return avgs

    # ========================= Internals =========================
    def _closest_goal(self, agent_rc):
        goals = self.simulator.goal_positions
        if not goals:
            return agent_rc
        dists = [cheb(agent_rc, g) for g in goals]
        return goals[int(np.argmin(dists))]

    def _closest_goal_dist(self, agent_rc):
        g = self._closest_goal(agent_rc)
        return cheb(agent_rc, g)

    def _goal_vec_norm(self, agent_rc):
        """Return (dx_norm, dy_norm, d_norm) toward the nearest goal, normalized by max(n_rows, n_cols)."""
        g = self._closest_goal(agent_rc)
        n = max(self.norm, 1.0)
        dx = float(g[0] - agent_rc[0]) / n
        dy = float(g[1] - agent_rc[1]) / n
        d = cheb(agent_rc, g) / n
        return dx, dy, min(1.0, d)

    def _battery_stats(self, sensor_batts):
        if not sensor_batts:
            return (0.0, 0.0, 0.0)
        vals = np.array(list(sensor_batts.values()), dtype=np.float32)
        return float(vals.min()), float(vals.mean()), float(vals.max())

    def _make_obs(self, agent_rc, sensor_batts):
        dx, dy, d = self._goal_vec_norm(agent_rc)
        bmin, bmean, bmax = self._battery_stats(sensor_batts)
        vec = np.array([dx, dy, d, bmin, bmean, bmax], dtype=np.float32)
        vec[0:2] = np.clip(vec[0:2], -1.0, 1.0)
        vec[2] = np.clip(vec[2], 0.0, 1.0)
        return {"vec": vec}

    def _build_batt_map_and_update_costs(self, state):
        """Build per-cell predicted battery map and mirror into cost_map via tiers."""
        assert self.cost_map is not None and self.current_start_xy is not None
        H, W = self.simulator.n_rows, self.simulator.n_cols
        c0, r0 = self.current_start_xy

        ys = np.arange(H)[:, None]
        xs = np.arange(W)[None, :]
        D = np.maximum(np.abs(xs - c0), np.abs(ys - r0)).astype(int)

        s_order = [pos for pos in self.simulator.sensor_positions]
        batts_norm = np.array([state["sensor_batteries"][pos] for pos in s_order], dtype=np.float32) / 100.0

        batt_map = np.zeros((H, W), dtype=np.float32)

        if self.mode == "constant_rate":
            rates = self.constant_rates
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                rate = float(rates.get(sensor, 0.0))
                curr = batts_norm[self.sensor_index[sensor]] * 100.0
                batt_map[y, x] = max(curr - rate * D[y, x], 0.0)
        else:  # 'static'
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                batt_map[y, x] = float(batts_norm[self.sensor_index[sensor]] * 100.0)

        # Reflect into cost_map & update D*
        dirty = []
        for y, x in self.simulator.free_cells:
            tier = sensor_cost_tier(batt_map[y, x])
            if self.cost_map[y, x] != tier:
                self.cost_map[y, x] = tier
                dirty.append((x, y))

        if self.use_planner_overlay and self.pathfinder is not None and dirty:
            for x, y in dirty:
                self.pathfinder.updateVertex(x, y)
                for nx, ny in self.pathfinder.neighbors(x, y):
                    self.pathfinder.updateVertex(nx, ny)

        return batt_map

    def _replan_from(self, agent_rc):
        if not (self.use_planner_overlay and self.pathfinder is not None):
            return
        cx, cy = agent_rc[1], agent_rc[0]
        self.pathfinder.updateStart(cx, cy)
        self.pathfinder.computeShortestPath()

    def _render_frame(self, batt_map):
        try:
            ok = self.simulator.render(
                show_miners=self.show_miners,
                dstar_path=self.pathfinder.getShortestPath() if (self.use_planner_overlay and self.pathfinder) else [],
                path_history=self.path_history,
                predicted_battery_map=batt_map if self.show_predicted else None
            )
            if not ok:
                self.render_enabled = False  # window closed by user
        except Exception:
            self.render_enabled = False

    def _current_cell_battery(self, state_dict, agent_rc):
        """
        Battery (float) for the sensor connected to the agent's current cell.
        agent_rc is (r, c); cell_to_sensor expects (x, y) == (c, r).
        """
        try:
            cell_xy = (agent_rc[1], agent_rc[0])  # (x, y)
            sensor_pos = self.simulator.cell_to_sensor.get(cell_xy)
            if sensor_pos is None:
                return None
            batts = state_dict.get("sensor_batteries", {})
            val = batts.get(sensor_pos)
            return float(val) if val is not None else None
        except Exception:
            return None
