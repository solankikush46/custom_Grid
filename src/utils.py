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

# e.g. "mine_50x50_12miners"
_EXP_RE = re.compile(r'^([A-Za-z0-9\-]+)_(\d+)x(\d+)_?(\d+)miners$')

def parse_experiment_folder(experiment_folder):
    """
    Parse names like 'mine_50x50_12miners' (underscore before 'miners' optional).

    Returns a dict (strings/ints/floats only):
        {
          "experiment_folder": "<input>",
          "prefix": "mine",
          "rows": 50,
          "cols": 50,
          "n_miners": 12,
          "size_token": "50x50",
          "miners_token": "12miners",
          "grid_stem": "mine_50x50",
          "grid_file": "mine_50x50.txt",
          "norm": 50.0
        }
    """
    if not isinstance(experiment_folder, str) or not experiment_folder:
        raise ValueError("experiment_folder must be a non-empty string")

    m = _EXP_RE.match(experiment_folder)
    if not m:
        raise ValueError(
            "Invalid experiment_folder format. Expected like 'mine_50x50_12miners'. "
            "Example prefixes are free-form letters/digits/dashes."
        )

    prefix, rows_s, cols_s, miners_s = m.groups()
    rows = int(rows_s)
    cols = int(cols_s)
    n_miners = int(miners_s)

    size_token = f"{rows}x{cols}"
    miners_token = f"{n_miners}miners"
    grid_stem = f"{prefix}_{size_token}"
    grid_file = f"{grid_stem}.txt"
    norm = float(max(rows, cols))

    return {
        "experiment_folder": experiment_folder,
        "prefix": prefix,
        "rows": rows,
        "cols": cols,
        "n_miners": n_miners,
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
    """
    if not os.path.isdir(exp_dir):
        return []

    out = []
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if not os.path.isdir(full):
            continue
        if not (name.startswith("PPO_") or name.startswith("RecurrentPPO_") or name.startswith("DQN_")):
            continue
        model_zip = os.path.join(full, "model.zip")
        try:
            mtime = os.path.getmtime(full)
        except Exception:
            mtime = 0
        out.append({
            "name": name,
            "path": full,
            "type": name.split("_", 1)[0],
            "has_model_zip": os.path.isfile(model_zip),
            "model_zip": model_zip if os.path.isfile(model_zip) else None,
            "modified": mtime,
            "modified_iso": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
        })

    out.sort(key=lambda d: d["modified"] or 0, reverse=True)  # newest first
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

def parse_experiment_data(experiment_folder):
    """
    Full parse of an experiment folder name.

    Returns a dict with:
      - "experiment": tokens and derived values
      - "grid": grid file info (path, existence, counts)
      - "paths": canonical SAVE_DIR paths for artifacts
      - "artifacts": presence and parsed contents of saved artifacts

    Example keys:
      data["experiment"]["n_miners"]
      data["grid"]["file"], data["grid"]["path"], data["grid"]["counts"]
      data["artifacts"]["avg_depletion_map"]  # {(r,c): avg_drop}
      data["artifacts"]["latest_run"]         # most recent PPO_* / DQN_* subdir
    """
    parts = parse_experiment_folder(experiment_folder)

    # Grid file resolution + scan
    grid_path = _find_grid_file(parts["grid_file"])
    grid_counts = _scan_grid_ascii(grid_path) if grid_path else {}

    # SAVE_DIR artifacts
    exp_dir = os.path.join(SAVE_DIR, experiment_folder)
    avg_path    = os.path.join(exp_dir, "avg_sensor_depletion.json")
    deltas_path = os.path.join(exp_dir, "battery_deltas.json")

    avg_raw    = _load_json_safe(avg_path, default=None)
    deltas_raw = _load_json_safe(deltas_path, default=None)

    avg_parsed = _parse_rc_keys(avg_raw) if isinstance(avg_raw, dict) else {}

    # For deltas we keep the raw JSON for transparency, but also expose a
    # convenience map of sums (optional).
    deltas_sum_map = {}
    if isinstance(deltas_raw, dict):
        for k, v in deltas_raw.items():
            try:
                s_val = float(v[0]) if isinstance(v, list) and len(v) >= 1 else 0.0
                r_s, c_s = str(k).split(",")
                deltas_sum_map[(int(r_s), int(c_s))] = s_val
            except Exception:
                continue

    runs = _list_model_runs(exp_dir)

    return {
        "experiment": {
            "name": parts["experiment_folder"],
            "prefix": parts["prefix"],
            "rows": parts["rows"],
            "cols": parts["cols"],
            "n_miners": parts["n_miners"],
            "norm": parts["norm"],
            "size_token": parts["size_token"],
            "miners_token": parts["miners_token"],
            "grid_stem": parts["grid_stem"],
            "grid_file": parts["grid_file"],
        },
        "grid": {
            "file": parts["grid_file"],
            "path": grid_path,
            "exists": bool(grid_path and os.path.isfile(grid_path)),
            "counts": grid_counts,
        },
        "paths": {
            "save_dir": exp_dir,
            "avg_depletion_json": avg_path,
            "battery_deltas_json": deltas_path,
        },
        "artifacts": {
            "has_avg_depletion": bool(avg_parsed),
            "has_battery_deltas": bool(deltas_raw),
            "avg_depletion_map": avg_parsed,     # {(r,c): avg_drop}
            "battery_deltas_raw": deltas_raw,    # {"r,c": [sum, count]}
            "battery_deltas_sum_map": deltas_sum_map,  # {(r,c): sum}
            "runs": runs,                         # newest-first list
            "latest_run": runs[0] if runs else None,
        }
    }

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
        if require_model and not os.path.isfile(os.path.join(run_dir, "model.zip")):
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
