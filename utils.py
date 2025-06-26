# utils.py

import os
import numpy as np
import math

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

def euclidean_distance(x0, x1, y0, y1):
    dx = x0 - x1
    dy = y0 - y1
    return math.sqrt(dx * dx + dy * dy)

# i should make it so normalize is not passed in,
# but w and h are set to none by default, and if they are both none
# then you aren't supposed to normalize, otherwise normalize
def euclidean_distances(pos, targets, grid_width, grid_height, normalize=True):
    x0, y0 = pos
    if normalize:
        norm = euclidean_distance(0, grid_width - 1, 0, grid_height - 1)
        return np.array([
            euclidean_distance(x0, tx, y0, ty) / norm for tx, ty in targets
        ], dtype=np.float32)
    else:
        return np.array([
            euclidean_distance(x0, tx, y0, ty) for tx, ty in targets
        ], dtype=np.float32)

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
