# utils.py

import os
import numpy as np
import math
import re
from datetime import datetime

from src.constants import *

##==============================================================
## Helpers
##==============================================================
def chebyshev_distance(x0, x1, y0, y1):
    dx = abs(x0 - x1)
    dy = abs(y0 - y1)
    return max(dx, dy)

def chebyshev_distances(pos, targets, grid_width, grid_height, normalize=True):
        x0, y0 = pos
        if normalize:
            norm = max(grid_width - 1, grid_height - 1)
            return np.array([
                chebyshev_distance(x0, tx, y0, ty) / norm for tx, ty in targets
            ], dtype=np.float32)
        else:
            return np.array([
                chebyshev_distance(x0, tx, y0, ty) for tx, ty in targets
            ], dtype=np.float32)

def euclidean_distance(pos1, pos2):
    """
    Calculates the Euclidean distance between two points using NumPy.

    This function is a more direct and efficient replacement for the original.
    It assumes pos1 and pos2 are array-like (e.g., tuples, lists, or NumPy arrays).

    Args:
        pos1 (array-like): The first point, e.g., (x1, y1).
        pos2 (array-like): The second point, e.g., (x2, y2).

    Returns:
        float: The Euclidean distance.
    """
    # np.linalg.norm calculates the length (L2 norm) of the vector difference,
    # which is the Euclidean distance.
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def euclidean_distances(pos, targets, grid_width=None, grid_height=None):
    """
    Calculates the Euclidean distance from a single position to an array of targets.
    This version is vectorized for high performance.

    Args:
        pos (array-like): The single starting point, e.g., (x, y).
        targets (array-like): A list or array of target points, e.g., [[x1, y1], [x2, y2], ...].
        grid_width (int, optional): If provided, used for normalization.
        grid_height (int, optional): If provided, used for normalization.

    Returns:
        np.ndarray: A 1D NumPy array of distances.
    """
    # Ensure inputs are NumPy arrays for vectorized operations
    pos_arr = np.array(pos)
    targets_arr = np.array(targets)

    # Check if there are any targets to prevent errors with empty arrays
    if targets_arr.shape[0] == 0:
        return np.array([], dtype=np.float32)

    # --- Vectorized Calculation ---
    distances = np.linalg.norm(targets_arr - pos_arr, axis=1)

    # --- Normalization ---
    if grid_width is not None and grid_height is not None:
        # The maximum possible distance is the diagonal of the grid
        diagonal_vector = np.array([grid_width - 1, grid_height - 1])
        max_dist = np.linalg.norm(diagonal_vector)
        
        # Avoid division by zero if the grid is just a single point
        if max_dist > 0:
            distances /= max_dist
            
    return distances.astype(np.float32)

def load_obstacles_from_file(filename="obstacle_coords.txt"):
    obstacles = []
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    r, c = map(int, line.split(","))
                    obstacles.append((r, c))
    except FileNotFoundError:
        print(f"[WARNING] Obstacle file '{filename}' not found.")
    return obstacles

def load_sensors_with_batteries(filename="sensor_coords.txt"):
    sensors = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                r, c, battery = map(float, line.strip().split(","))
                sensors[(int(r), int(c))] = battery
    except FileNotFoundError:
        print(f"[WARNING] Sensor file '{filename}' not found.")
    return sensors

def get_c8_neighbors_status(grid, agent_pos, obstacle_val= ('#', 'S', 'B')):
    """
    Returns a list of 8 values for each direction: [N, NE, E, SE, S, SW, W, NW]
    1 if blocked, 0 if free.
    """
    directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    x, y = agent_pos
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            neighbors.append(1 if grid[nx, ny] == obstacle_val else 0)
        else:
            neighbors.append(1)  # treat out-of-bounds as blocked
    return neighbors

def make_agent_feature_matrix(agent_pos, neighbors, last_action, goal_dist, sensor_batteries, max_sensors):
    """
    Build a 5x5 feature matrix for the agent observation, accommodating up to 9 sensors.
    """
    feature_matrix = np.zeros((5, 5), dtype=np.float32)
    # Row 0: agent info
    feature_matrix[0, :5] = [
        agent_pos[0], agent_pos[1], last_action, goal_dist, 0.0
    ]
    # Row 1: neighbors [N, NE, E, SE, S]
    feature_matrix[1, :5] = neighbors[:5]
    # Row 2: [SW, W, NW, 0, 0]
    feature_matrix[2, :3] = neighbors[5:8]
    # Row 3 and 4: sensor batteries (up to 9 sensors)
    nb = len(sensor_batteries)
    feature_matrix[3, :5] = sensor_batteries[:5]
    feature_matrix[4, :4] = sensor_batteries[5:9]
    return feature_matrix

# ------------------------------------------------------------------------------
# Experiment name parsing
# ------------------------------------------------------------------------------
def parse_experiment_folder(experiment_folder):
    """
    Parse names like:
      - New: "mine_50x50__12miners__mlp__reward_d"
      - Old: "mine_50x50_12miners_mlp"  (reward defaults to "reward_d")

    Returns a dict with simple types only.
    """
    if not isinstance(experiment_folder, str) or not experiment_folder.strip():
        raise ValueError("experiment_folder must be a non-empty string")

    # Reuse the canonical parser so naming stays consistent
    meta = get_metadata(experiment_folder)  # -> {"grid","miners","arch","reward"}
    grid = meta["grid"]
    miners = int(meta["miners"])
    arch = meta["arch"]
    reward = meta.get("reward", "reward_d")

    # Extract rows/cols from the grid token without regex, e.g. "mine_50x50"
    rows = cols = None
    for token in reversed(grid.split("_")):
        if "x" in token:
            left, right = token.split("x", 1)
            if left.isdigit() and right.isdigit():
                rows, cols = int(left), int(right)
                break
    if rows is None or cols is None:
        raise ValueError(f"Cannot find '<rows>x<cols>' in grid token: {grid!r}")

    size_token = f"{rows}x{cols}"
    miners_token = f"{miners}miners"
    grid_stem = grid
    grid_file = f"{grid_stem}.txt"
    norm = float(max(rows, cols))

    return {
        "experiment_folder": experiment_folder.strip(),
        "grid": grid,
        "arch": arch,
        "reward": reward,
        "rows": rows,
        "cols": cols,
        "n_miners": miners,
        "size_token": size_token,
        "miners_token": miners_token,
        "grid_stem": grid_stem,
        "grid_file": grid_file,
        "norm": norm,
    }

# ------------------------------------------------------------------------------
# Grid discovery / scan
# ------------------------------------------------------------------------------

def _find_grid_file(grid_file):
    """
    Try likely locations for the ASCII grid. Returns a path or None.
    Search order: FIXED_GRID_DIR, RANDOM_GRID_DIR, GRID_DIR (root).
    """
    candidates = [
        os.path.join(FIXED_GRID_DIR, grid_file),
        os.path.join(RANDOM_GRID_DIR, grid_file),
        os.path.join(GRID_DIR, grid_file),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _scan_grid_ascii(path):
    """
    Read an ASCII grid file and count key symbols.
    Returns:
        {
          "sensors": int, "goals": int, "obstacles": int, "bases": int,
          "miners_in_grid": int, "empty": int, "total_cells": int,
          "rows": int, "cols": int
        }
    """
    counts = {
        "sensors": 0,
        "goals": 0,
        "obstacles": 0,
        "bases": 0,
        "miners_in_grid": 0,
        "empty": 0,
        "total_cells": 0,
        "rows": 0,
        "cols": 0,
    }
    if not path or not os.path.isfile(path):
        return counts

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        return counts

    counts["rows"] = len(lines)
    counts["cols"] = max(len(line) for line in lines)

    for line in lines:
        for ch in line:
            if ch == SENSOR_CHAR:
                counts["sensors"] += 1
            elif ch == GOAL_CHAR:
                counts["goals"] += 1
            elif ch == OBSTACLE_CHAR:
                counts["obstacles"] += 1
            elif ch == BASE_STATION_CHAR:
                counts["bases"] += 1
            elif ch == MINER_CHAR:
                counts["miners_in_grid"] += 1
            else:
                counts["empty"] += 1
            counts["total_cells"] += 1

    return counts

# ------------------------------------------------------------------------------
# SAVE_DIR inspection
# ------------------------------------------------------------------------------

def _list_model_runs(exp_dir):
    """
    Find subdirs in SAVE_DIR/<exp> that look like model runs (PPO_*, RecurrentPPO_*, DQN_*).
    Returns a sorted list (newest first) of dicts with basic metadata.

    For each run dir, we accept ANY model artifact matching "model*.zip" and pick a preferred one:
      1) model.zip
      2) model_att.zip
      3) model_cnn.zip
      4) else the first available model*.zip
    """
    if not os.path.isdir(exp_dir):
        return []

    out = []
    try:
        run_names = os.listdir(exp_dir)
    except Exception:
        run_names = []

    for name in run_names:
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        if not (name.startswith("PPO_") or name.startswith("RecurrentPPO_") or name.startswith("DQN_")):
            continue

        # Collect all model*.zip files in the run directory
        try:
            files = os.listdir(full)
        except Exception:
            files = []

        model_candidates = [os.path.join(full, fn) for fn in files
                            if fn.startswith("model") and fn.endswith(".zip")]
        has_any_model = len(model_candidates) > 0

        # Choose a single representative model_zip to surface (preference order)
        model_zip = None
        if has_any_model:
            prefer = ("model.zip", "model_att.zip", "model_cnn.zip")
            by_name = {os.path.basename(p): p for p in model_candidates}
            for pref in prefer:
                if pref in by_name:
                    model_zip = by_name[pref]
                    break
            if model_zip is None:
                # Fall back to the first candidate (deterministic order)
                model_candidates.sort()
                model_zip = model_candidates[0]

        try:
            mtime = os.path.getmtime(full)
        except Exception:
            mtime = 0

        out.append({
            "name": name,
            "path": full,
            "type": name.split("_", 1)[0],
            "has_model_zip": has_any_model,
            "model_zip": model_zip,
            "modified": mtime,
            "modified_iso": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
        })

    # newest first
    out.sort(key=lambda d: d["modified"] or 0, reverse=True)
    return out

def _load_json_safe(path, default=None):
    if default is None:
        default = {}
    try:
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _parse_rc_keys(maybe_rc_map):
    """
    Convert {"r,c": value} -> {(r,c): float(value)}. Ignore malformed keys.
    """
    parsed = {}
    if not isinstance(maybe_rc_map, dict):
        return parsed
    for k, v in maybe_rc_map.items():
        try:
            r_s, c_s = str(k).split(",")
            parsed[(int(r_s), int(c_s))] = float(v)
        except Exception:
            continue
    return parsed

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
_PPO_RE = re.compile(r"^PPO_(\d+)$")

def latest_ppo_run(experiment_folder, require_model=True):
    """
    Return (run_dir, run_name) for the highest-numbered PPO_<n> under
    SAVE_DIR/<experiment_folder>. If require_model=True, only consider
    runs that contain model.zip. Tiebreak by modification time.
    """
    base = os.path.join(SAVE_DIR, experiment_folder)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Experiment folder not found: {base}")

    best = None  # (n, mtime, run_name, run_dir)
    for name in os.listdir(base):
        m = _PPO_RE.match(name)
        if not m:
            continue
        run_dir = os.path.join(base, name)
        if not os.path.isdir(run_dir):
            continue
        if require_model:
            has_any_model = any(fn.startswith("model") and fn.endswith(".zip")
                                for fn in os.listdir(run_dir))
            if not has_any_model:
                continue
        n = int(m.group(1))
        mt = os.path.getmtime(run_dir)
        if best is None or (n, mt) > (best[0], best[1]):
            best = (n, mt, name, run_dir)

    if best is None:
        raise FileNotFoundError(
            f"No PPO_<n> runs found under {base}"
            + (" with model.zip" if require_model else "")
        )

    return best[3], best[2]  # (run_dir, run_name)

def latest_ppo_model_path(experiment_folder):
    """
    Return the absolute path to model.zip in the latest PPO_<n> run.
    """
    run_dir, _ = latest_ppo_run(experiment_folder, require_model=True)
    return os.path.join(run_dir, "model.zip")

# ==============================================================
# Metadata helpers  (now encode reward + use '__' between parts)
# ==============================================================
def make_experiment_name(metadata: dict) -> str:
    """
    Build an experiment string from metadata.

    Required keys:
        grid:   "mine_50x50" or "50x50"
        miners: 12 (int) or "12" or "12miners"
        arch:   "cnn" | "attn" | "mlp" | <str>
    Optional:
        reward: "reward_d" (default) or any key in the reward registry

    Returns (double-underscore separators):
        e.g. "mine_50x50__12miners__cnn__reward_d"
    """
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dict")

    for k in ("grid", "miners", "arch"):
        if k not in metadata:
            raise ValueError("metadata must contain 'grid', 'miners', and 'arch'")

    grid = str(metadata["grid"]).strip()
    if not grid:
        raise ValueError("metadata['grid'] must be a non-empty string")

    miners_val = metadata["miners"]
    if isinstance(miners_val, str):
        m = miners_val.strip()
        if m.endswith("miners"):
            m = m[:-6]
        if not m.isdigit():
            raise ValueError(f"metadata['miners'] must be an int or digit string, got: {miners_val!r}")
        miners = int(m)
    elif isinstance(miners_val, int):
        miners = miners_val
    else:
        raise ValueError("metadata['miners'] must be an int or a string")

    if miners <= 0:
        raise ValueError("metadata['miners'] must be a positive integer")

    arch = str(metadata["arch"]).strip()
    if not arch:
        raise ValueError("metadata['arch'] must be a non-empty string")

    reward_key = str(metadata.get("reward", "reward_d")).strip() or "reward_d"

    # double-underscore between major parts
    return f"{grid}__{miners}miners__{arch}__{reward_key}"


def get_metadata(experiment_name: str) -> dict:
    """
    Inverse of make_experiment_name.

    Accepts (new format with '__' between parts):
        "mine_50x50__12miners__cnn__reward_d"
        "50x50__12miners__mlp__reward_d"
    Backward-compatible (old '_' format without reward):
        "mine_50x50_12miners_cnn"  -> reward defaults to "reward_d"

    Returns:
        {"grid": <str>, "miners": <int>, "arch": <str>, "reward": <str>}
    """
    if not isinstance(experiment_name, str):
        raise ValueError("experiment_name must be a string")

    name = experiment_name.strip()
    if "__" in name:
        parts = name.split("__")
        if len(parts) < 4:
            raise ValueError(f"Bad experiment name (need 4 parts w/ '__'): {experiment_name!r}")
        grid = parts[0].strip()
        miners_tok = parts[1].strip()
        arch = parts[2].strip()
        reward_key = parts[3].strip()
    else:
        # Back-compat for old names like "mine_50x50_12miners_cnn"
        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Bad experiment name (need at least 3 parts): {experiment_name!r}")
        grid = "_".join(parts[:-2]).strip()
        miners_tok = parts[-2].strip()
        arch = parts[-1].strip()
        reward_key = "reward_d"

    if not grid:
        raise ValueError(f"Cannot parse grid from: {experiment_name!r}")
    if not miners_tok.endswith("miners"):
        raise ValueError(f"Bad miners token (must end with 'miners'): {miners_tok!r}")
    num_str = miners_tok[:-6]
    if not num_str.isdigit():
        raise ValueError(f"Cannot parse miners count from: {miners_tok!r}")
    miners = int(num_str)
    if miners <= 0:
        raise ValueError("miners must be a positive integer")
    if not arch:
        raise ValueError(f"Cannot parse arch from: {experiment_name!r}")
    if not reward_key:
        reward_key = "reward_d"

    return {"grid": grid, "miners": miners, "arch": arch, "reward": reward_key}

def parse_experiment_data(experiment_folder):
    """
    Minimal parser for MineEnv needs.

    Supports names like:
      - "mine_50x50__20miners__mlp__reward_d" (new, with '__')
      - "mine_50x50_20miners_mlp"             (legacy, with '_')

    Returns ONLY the keys MineEnv reads:
      {
        "experiment": {"n_miners": <int>, "norm": <float>},
        "grid":       {"file": "<grid>.txt"},
        "paths":      {"avg_depletion_json": "<abs path>"},
        "artifacts":  {}   # (empty; MineEnv will fall back to disk if needed)
      }
    """
    if not isinstance(experiment_folder, str) or not experiment_folder.strip():
        raise ValueError("experiment_folder must be a non-empty string")

    name = experiment_folder.strip()

    # -------- pick out major tokens (grid, miners) without regex --------
    if "__" in name:
        parts = [p.strip() for p in name.split("__") if p.strip()]
        if len(parts) < 2:
            raise ValueError("bad experiment name (need at least grid and miners tokens)")
        grid_token = parts[0]                     # e.g. "mine_50x50"
        miners_tok = parts[1]                     # e.g. "20miners"
    else:
        parts = [p.strip() for p in name.split("_") if p.strip()]
        if len(parts) < 3:
            raise ValueError("bad experiment name (legacy format needs >= 3 tokens)")
        grid_token = "_".join(parts[:-2])         # join to preserve underscores in grid
        miners_tok = parts[-2]                    # e.g. "20miners"

    if not miners_tok.endswith("miners"):
        raise ValueError(f"bad miners token: {miners_tok!r}")
    miners_str = miners_tok[:-6]
    if not miners_str.isdigit():
        raise ValueError(f"cannot parse miners count from: {miners_tok!r}")
    n_miners = int(miners_str)

    # -------- extract rows/cols from grid token (search last 'RxC' piece) --------
    rows = cols = None
    for tok in reversed(grid_token.split("_")):
        if "x" in tok:
            left, right = tok.split("x", 1)
            if left.isdigit() and right.isdigit():
                rows, cols = int(left), int(right)
                break
    if rows is None or cols is None:
        raise ValueError(f"cannot find '<rows>x<cols>' in grid token: {grid_token!r}")

    norm = float(max(rows, cols))
    grid_file = f"{grid_token}.txt"

    # -------- paths used by MineEnv (for avg depletion) --------
    from .constants import SAVE_DIR
    base_dir = os.path.join(SAVE_DIR, name)
    avg_json = os.path.join(base_dir, "avg_sensor_depletion.json")

    # Return only what MineEnv reads
    return {
        "experiment": {
            "n_miners": n_miners,
            "norm": norm,
        },
        "grid": {
            "file": grid_file,
        },
        "paths": {
            "avg_depletion_json": avg_json,
        },
        "artifacts": {},  # keep empty; MineEnv will read avg json if present
    }
