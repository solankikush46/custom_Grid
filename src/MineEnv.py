# MineEnv.py

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .DStarLite.DStarLite import DStarLite
from .MineSimulator import MineSimulator
from .constants import *
from .utils import *
from .train import *
from .reward_functions import *  

# ===============================
# Helpers
# ===============================
def cheb(a, b):
    """Chebyshev distance between two (r, c) cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def sensor_cost_tier(batt):
    """Map raw battery (0–100) to movement penalty via COST_TABLE with clamping."""
    return COST_TABLE[int(batt)]

def _sorted_allowed_moves():
    """
    Build the discrete action set directly from MOVE_TO_ACTION_MAP keys
    (keeps you in sync with constants.DIRECTION_MAP). Filters None-mapped moves.
    """
    items = [(drc, tok) for drc, tok in MOVE_TO_ACTION_MAP.items() if tok is not None]
    items.sort(key=lambda it: (it[0][0], it[0][1]))  # stable order
    return [drc for drc, _ in items]

#================================
# MineEnv
#================================
class MineEnv(gym.Env):
    """
    Gym environment integrating MineSimulator with a D* Lite cost map derived from sensor batteries.

    Modes:
      - 'constant_rate' : uses per-sensor average depletion rates collected from runs
                          (avg_sensor_depletion.json under SAVE_DIR/<exp>/).
      - 'static'        : uses a single global predicted depletion rate (the mean of the
                          collected per-sensor averages) applied to all sensors equally.

    Notes:
      - No step-based truncation; episodes end only when the guided miner reaches any goal.
      - D* is optional and used only for on-screen overlay; RL chooses actions.
    """

    metadata = {"render.modes": ["human", "none"]}
    # if statements are slow, so we set the observation function based on arch in __init__
    def __init__(
        self,
        experiment_folder,
        render=False,
        show_miners=False,
        show_predicted=True,
        mode="static",              # 'static' | 'constant_rate'
        use_planner_overlay=True,
        arch: str = "mlp",
        reward_fn=reward_d,         # default hook (can pass any reward_* callable)
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
        self.use_planner_overlay = bool(use_planner_overlay)
        self.arch = arch
        self.is_attn = "a" in self.arch
        self.is_cnn = self.is_attn or "c" in self.arch
        self.reward_fn = reward_fn if callable(reward_fn) else reward_d

        # --- manual depletion collection state (optional external use) ---
        self._delta_sums = {}
        self._delta_counts = {}
        self._prev_sensor_batts = {}

        render_mode = 'human' if self.render_enabled else None

        # Simulator
        self.simulator = MineSimulator(
            grid_file=grid_file,
            n_miners=n_miners,
            render_mode=render_mode,
            show_predicted=self.show_predicted,
        )

        H = self.simulator.n_rows
        W = self.simulator.n_cols

        # Stable sensor ordering
        self.sensor_index = {pos: i for i, pos in enumerate(self.simulator.sensor_positions)}

        # ================== Depletion rate sources ==================
        self.constant_rates = {}   # {(r,c): rate}
        self.global_rate = None     # mean rate (for 'static')

        if arts.get("avg_depletion_map"):
            self.constant_rates = dict(arts["avg_depletion_map"])
            if "avg_depletion_mean" in arts and isinstance(arts["avg_depletion_mean"], (int, float)):
                self.global_rate = float(arts["avg_depletion_mean"])
            else:
                vals = list(self.constant_rates.values())
                self.global_rate = float(np.mean(vals)) if len(vals) > 0 else 0.0
        else:
            avg_json = paths.get("avg_depletion_json")
            if avg_json and os.path.isfile(avg_json):
                with open(avg_json, "r") as f:
                    data_json = json.load(f)
                for k, v in data_json.items():
                    r, c = map(int, k.split(","))  # keys like "r,c"
                    self.constant_rates[(r, c)] = float(v)
                vals = list(self.constant_rates.values())
                self.global_rate = float(np.mean(vals)) if len(vals) > 0 else 0.0
            else:
                self.constant_rates = {}
                self.global_rate = 0.76
        # ============================================================

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

        # Action/Observation spaces
        self._ACTIONS = list(ACTIONS)
        self.action_space = spaces.Discrete(len(self._ACTIONS))

        if self.is_cnn:
            if self.arch[-1] == "0":
                # 5 channels: agent, obstacle|base, sensor, sensor_batt_norm, goal
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(5, H, W), dtype=np.float32
                )
            elif self.arch[-1] == "1":
                # 4 channels: [can_move, pred_batt_norm, is_agent_pos, is_on_dstar_path]
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, shape=(4, H, W), dtype=np.float32
                )
        else: # if arch="mlp"
            self.observation_space = spaces.Box(
                low=np.array([-1, -1, 0, 0, 0, 0], dtype=np.float32),
                high=np.array([ 1,  1, 1, 100, 100, 100], dtype=np.float32),
                dtype=np.float32,
                shape=(6,),
            )

        # Choose observation function to avoid per-step branching
        if self.is_cnn:
            if self.arch.endswith("1"):
                self._get_obs = self._make_obs_cnn1
            else:
                # default to 5-channel (arch "0" or any other CNN)
                self._get_obs = self._make_obs_cnn0
        else:
            self._get_obs = self._make_obs

        # Exposed value for reward adapters (old API compatibility)
        self.current_battery_level = None
        self._agent_rc_after_step = None

        # cache for latest predicted battery map
        self.batt_map = None

    # ========================= Gym API =========================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Episode stats
        self._steps = 0
        self._cumulative_reward = 0.0
        self._visited.clear()
        self._revisit_count = 0
        self._obstacle_hits = 0

        # Reset simulator
        state = self.simulator.reset()
        agent_rc = tuple(state["guided_miner_pos"])
        self._agent_rc_after_step = agent_rc
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
        self.batt_map = self._build_batt_map_and_update_costs(state)
        if self.use_planner_overlay and self.pathfinder is not None:
            self._replan_from(agent_rc)

        # Set current battery for adapters/info
        self.current_battery_level = self._current_cell_battery(state, agent_rc)

        # Observation
        obs = self._get_obs(agent_rc, state.get("sensor_batteries", {}))

        # Distance to nearest goal for info
        d0 = self._closest_goal_dist(agent_rc)
        info = {
            "sensor_batteries": state.get("sensor_batteries", {}),
            "distance_to_goal": float(d0),
            "current_battery": float(self.current_battery_level) if self.current_battery_level is not None else None,
            "terminated": False,
            "truncated": False,
        }

        if self.render_enabled and self.simulator.render_mode == "human":
            self._render_frame(self.batt_map)

        return obs, info

    def step(self, action):
        if self.cost_map is None or self.current_start_xy is None:
            obs, info = self.reset()
            return obs, 0.0, False, False, info

        self._steps += 1

        # ---- decode action & compute intended new position ----
        dr, dc = self._ACTIONS[int(action)]
        sim_token = MOVE_TO_ACTION_MAP.get((dr, dc), None)

        prev_rc = tuple(self.simulator.guided_miner_pos)
        new_pos_intended = (prev_rc[0] + dr, prev_rc[1] + dc)

        # ---- step simulator ----
        try:
            s = self.simulator.step(guided_miner_action=sim_token)
        except Exception as e:
            agent_rc = tuple(self.simulator.guided_miner_pos)
            obs = self._get_obs(agent_rc, {})
            info = {"error": str(e), "terminated": True, "truncated": False}
            return obs, -5.0, True, False, info

        agent_rc = tuple(s["guided_miner_pos"])
        self._agent_rc_after_step = agent_rc
        self.path_history.append(agent_rc)

        # Track obstacle bumps (for stats)
        if s.get("obstacle_hit", False):
            self._obstacle_hits += 1

        # Current cell's sensor battery, for reward adapters & info
        self.current_battery_level = self._current_cell_battery(s, agent_rc)

        # Update cost map and (optional) planner after movement
        self.current_start_xy = (agent_rc[1], agent_rc[0])  # (c, r)
        self.batt_map = self._build_batt_map_and_update_costs(s)
        if self.use_planner_overlay and self.pathfinder is not None:
            self._replan_from(agent_rc)

        # Compute distance to goal for info
        d_now = self._closest_goal_dist(agent_rc)

        # Compute reward (before updating visited to preserve old semantics)
        reward, sub = compute_reward(self, self.reward_fn, new_pos=new_pos_intended)

        # Update visited set and revisit count
        if agent_rc in self._visited:
            self._revisit_count += 1
        self._visited.add(new_pos_intended)

        # Episode termination conditions
        terminated = bool(agent_rc in self.simulator.goal_positions)
        truncated = False
        self._cumulative_reward += float(reward)

        if self.render_enabled and self.simulator.render_mode == "human":
            self._render_frame(self.batt_map)

        # Observation
        obs = self._get_obs(agent_rc, s.get("sensor_batteries", {}))

        info = {
            "sensor_batteries": s.get("sensor_batteries", {}),
            "current_reward": float(reward),
            "distance_to_goal": float(d_now),
            "current_battery": float(self.current_battery_level) if self.current_battery_level is not None else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "subrewards": sub,
        }
        if terminated or truncated:
            sbs = s.get("sensor_batteries", {})
            avg_batt = float(np.mean(list(sbs.values()))) if sbs else 0.0
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
        return None

    def close(self):
        self.simulator.close()

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
        return vec  # flat observation (shape (6,), float32)

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
            # Per-sensor collected averages
            rates = self.constant_rates
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                rate = float(rates.get(sensor, self.global_rate))  # fallback to global mean if missing
                curr = batts_norm[self.sensor_index[sensor]] * 100.0
                batt_map[y, x] = max(curr - rate * D[y, x], 0.0)
        else:
            # STATIC: same predicted depletion rate for all sensors (global mean)
            g_rate = float(self.global_rate)
            for y, x in self.simulator.free_cells:
                sensor = self.simulator.cell_to_sensor[(x, y)]
                curr = batts_norm[self.sensor_index[sensor]] * 100.0
                batt_map[y, x] = max(curr - g_rate * D[y, x], 0.0)

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

    def _make_obs_cnn0(self, agent_rc, sensor_batteries: dict) -> np.ndarray:
        """
        Build (5, H, W) observation with binary indicator maps plus a battery map.
        Channels:
        0: agent presence (0/1)
        1: obstacle OR base-station presence (0/1)
        2: sensor presence (0/1)
        3: sensor battery level (0..1) at sensor cells else 0
        4: goal presence (0/1)
        """
        H, W = self.simulator.n_rows, self.simulator.n_cols
        obs = np.zeros((5, H, W), dtype=np.float32)

        # 0) agent position
        ar, ac = agent_rc
        obs[0, ar, ac] = 1.0

        # 1) obstacles or base stations
        static = self.simulator.static_grid
        obs[1, static == OBSTACLE_ID] = 1.0
        for (r, c) in self.simulator.base_station_positions:
            obs[1, r, c] = 1.0

        # 2) sensor presence and 3) sensor battery (normalized)
        for pos in self.simulator.sensor_positions:
            r, c = pos
            obs[2, r, c] = 1.0
            batt = float(sensor_batteries.get(pos, 0.0)) / 100.0
            obs[3, r, c] = np.clip(batt, 0.0, 1.0)

        # 4) goals
        for (r, c) in self.simulator.goal_positions:
            obs[4, r, c] = 1.0

        return obs

    def _make_obs_cnn1(self, agent_rc, sensor_batteries: dict) -> np.ndarray:
        """
        Build (4, H, W) observation for alternative CNN architecture.
        Channels:
        0: can_move (1 if free cell, 0 if obstacle)
        1: predicted battery level (normalized 0..1)
        2: agent presence (0/1)
        3: D* Lite path (0/1)
        """
        H, W = self.simulator.n_rows, self.simulator.n_cols
        obs = np.zeros((4, H, W), dtype=np.float32)

        # 0) can_move: free cells (including base stations and sensor locations) marked 1, obstacles 0
        obs[0, :, :] = 0.0
        for (y, x) in self.simulator.free_cells:
            obs[0, y, x] = 1.0

        # 1) predicted battery map (normalized)
        if self.batt_map is not None:
            obs[1, :, :] = np.clip(self.batt_map / 100.0, 0.0, 1.0)
        else:
            obs[1, :, :] = 0.0

        # 2) agent position
        ar, ac = agent_rc
        obs[2, ar, ac] = 1.0

        # 3) current D* Lite shortest path
        path = self.pathfinder.getShortestPath() # what's the point of use_planner_overlay?
        for (x, y) in path:
            obs[3, y, x] = 1.0

        return obs

    # ========================= Old-API adapters for rewards =========================
    def get_goal_positions(self):
        """
        Return an iterable of goal cells as (r, c) pairs.
        """
        return set(self.simulator.goal_positions)

    def get_visited(self):
        """
        Return the set of visited (r, c) cells.
        """
        return self._visited

    def can_move_to(self, pos_rc) -> bool:
        """Old API name; use simulator’s validity check for the guided miner."""
        return bool(self.simulator.is_valid_guided_miner_move(pos_rc))

    def _compute_min_distance_to_goal(self) -> float:
        """
        Old API name; return normalized Euclidean distance to the nearest goal,
        computed from the *post-step* agent position (matches prior behavior).
        """
        r, c = self._agent_rc_after_step if self._agent_rc_after_step is not None else tuple(self.simulator.guided_miner_pos)
        best = float("inf")
        for (gr, gc) in self.simulator.goal_positions:
            d = np.hypot(gr - r, gc - c)
            if d < best:
                best = d
        norm = max(float(self.norm), 1.0)
        return float(best / norm)
