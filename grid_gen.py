import random
import os

def get_safe_zone_around(coords, rows, cols, radius=2):
    """
    Returns a set of coordinates within `radius` of each point in coords,
    ensuring all are within grid bounds.
    """
    safe_zone = set()
    for x, y in coords:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    safe_zone.add((nx, ny))
    return safe_zone

def generate_and_save_obstacles(rows, cols, n_obstacles=None, exclude_coords=[], filename="obstacle_coords.txt"):
    """
    Generate a set of obstacles while excluding certain zones (e.g., around goals).
    Writes coordinates to a text file.
    """
    if n_obstacles is None:
        n_obstacles = int(0.3 * rows * cols)

    coords = set()
    exclude_zone = get_safe_zone_around(exclude_coords, rows, cols, radius=2)

    attempts = 0
    max_attempts = n_obstacles * 10

    while len(coords) < n_obstacles and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in exclude_zone:
            coords.add((r, c))
        attempts += 1

    with open(filename, "w") as f:
        for r, c in coords:
            f.write(f"{r},{c}\n")

    print(f"[INFO] {len(coords)} obstacles saved to {filename}.")

def generate_and_save_sensors(rows, cols, n_sensors, obstacle_file="obstacle_coords.txt", sensor_file="sensor_coords.txt", goal_coords=[]):
    """
    Randomly generate sensor positions and battery levels.
    Ensures sensors do not overlap with obstacles or goals.
    """
    if not os.path.exists(obstacle_file):
        raise FileNotFoundError(f"Obstacle file '{obstacle_file}' not found.")

    obstacles = set()
    with open(obstacle_file, "r") as f:
        for line in f:
            r, c = map(int, line.strip().split(","))
            obstacles.add((r, c))

    forbidden = obstacles.union(goal_coords)
    sensors = set()
    sensor_data = []

    attempts = 0
    max_attempts = n_sensors * 20

    while len(sensors) < n_sensors and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in sensors and (r, c) not in forbidden:
            battery = random.randint(1, 100)
            sensor_data.append((r, c, battery))
            sensors.add((r, c))
        attempts += 1

    with open(sensor_file, "w") as f:
        for r, c, battery in sensor_data:
            f.write(f"{r},{c},{battery}\n")

    print(f"[INFO] {len(sensor_data)} sensors saved to {sensor_file}.")

def compute_sensor_radar_zone(sensor_coords, rows, cols, radius=2):
    """
    Return set of all grid cells within radar radius of sensors.
    """
    radar_zone = set()
    for x, y in sensor_coords:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    radar_zone.add((nx, ny))
    return radar_zone
