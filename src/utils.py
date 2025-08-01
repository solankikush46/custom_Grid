# utils.py

import os
import numpy as np
import math
import re

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

def extract_grid_hw_from_filename(grid_file):
    # Match patterns like mine_100x100.txt or mine_30x30.txt
    m = re.search(r'_(\d+)x(\d+)', grid_file)
    if m:
        height, width = int(m.group(1)), int(m.group(2))
        return height, width
    # Default/fallback
    return 100, 100