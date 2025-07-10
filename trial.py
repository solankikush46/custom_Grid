import numpy as np
import matplotlib.pyplot as plt
import heapq

# Directions: 8-way
DIRECTIONS = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]

def cell_quality_score(env, r, c):
    """Quality = (closer to goal) + (higher battery)"""
    if not env.goal_positions:
        dist_score = 0
    else:
        min_dist = min([abs(r - gr) + abs(c - gc) for (gr, gc) in env.goal_positions])
        # normalized: 0 (far) ... 1 (at goal)
        dist_score = 1.0 - (min_dist / (env.n_rows + env.n_cols))
    # Battery at nearest sensor (normalize)
    if env.sensor_batteries:
        closest_sensor = min(
            env.sensor_batteries.keys(),
            key=lambda pos: abs(pos[0] - r) + abs(pos[1] - c)
        )
        battery_score = env.sensor_batteries[closest_sensor] / 100.0
    else:
        battery_score = 0.5
    # Weighted sum: tweak as needed
    return 0.7 * dist_score + 0.3 * battery_score

def make_quality_grid(env):
    qg = np.zeros((env.n_rows, env.n_cols))
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.static_grid[r, c] == '#':   # Adjust if OBSTACLE constant is different
                qg[r, c] = -np.inf
            else:
                qg[r, c] = cell_quality_score(env, r, c)
    return qg

def find_highest_quality_path(env, quality_grid):
    """Finds the path from start to any goal maximizing total quality."""
    start = tuple(env.agent_pos)
    goals = set(env.goal_positions)
    n_rows, n_cols = env.n_rows, env.n_cols

    # heapq: (negative accumulated quality, path as list, current position)
    pq = [(-quality_grid[start[0], start[1]], [start], start)]
    best_total = -np.inf
    best_path = []
    visited = set()

    while pq:
        neg_accum, path, pos = heapq.heappop(pq)
        if pos in goals:
            total = -neg_accum
            if total > best_total:
                best_total = total
                best_path = path
            continue
        if pos in visited:
            continue
        visited.add(pos)
        for dr, dc in DIRECTIONS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                if env.static_grid[nr, nc] != '#' and (nr, nc) not in path:
                    heapq.heappush(
                        pq, (
                            neg_accum - quality_grid[nr, nc],
                            path + [(nr, nc)],
                            (nr, nc)
                        )
                    )
    return best_path, best_total

def plot_path_on_grid(env, path, quality_grid=None):
    """Visualize grid, path, agent, goals, and sensors with matplotlib."""
    grid_img = np.ones((env.n_rows, env.n_cols, 3))  # white

    # Color obstacles black
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.static_grid[r, c] == '#':
                grid_img[r, c] = [0,0,0]
    # Sensors: blue
    for (r,c) in getattr(env, 'sensor_batteries', {}):
        grid_img[r,c] = [0.4,0.6,1.0]
    # Goals: green
    for (r,c) in getattr(env, 'goal_positions', []):
        grid_img[r,c] = [0.3,1.0,0.3]
    # Path: red line
    if path:
        path_coords = np.array(path)
        plt.plot(path_coords[:,1], path_coords[:,0], color='red', linewidth=2, label='Best Path')
        for (r, c) in path:
            grid_img[r,c] = [1,0.5,0.5]
    # Start (agent): orange
    ar, ac = env.agent_pos
    grid_img[ar, ac] = [1.0, 0.6, 0.1]
    plt.imshow(grid_img, interpolation='nearest')
    plt.title('Optimal Quality Path (red), Agent (orange), Goals (green), Sensors (blue)')
    plt.axis('off')
    plt.legend()
    if quality_grid is not None:
        plt.figure()
        plt.imshow(quality_grid, cmap='viridis')
        plt.title('Cell Quality Score')
        plt.colorbar()
    plt.show()

# ---- Example usage ----
if __name__ == "__main__":
    # Instantiate your environment. E.g.,
    # env = GridWorldEnv(grid_height=10, grid_width=10, n_sensors=5, obstacle_percentage=0.15)
    # Or however you usually create it:
    env = GridWorldEnv(grid_height=10, grid_width=10, obstacle_percentage=0.15, n_sensors=4)
    obs, info = env.reset()

    quality_grid = make_quality_grid(env)
    best_path, best_score = find_highest_quality_path(env, quality_grid)

    print(f"Best Quality Path Score: {best_score:.2f}")
    print(f"Best Path (coords): {best_path}")

    plot_path_on_grid(env, best_path, quality_grid)
