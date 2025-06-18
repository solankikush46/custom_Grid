from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random 
import os  
import pygame
import setup_obstacles

if not os.path.exists("obstacle_coords.txt"):
    setup_obstacles.generate_and_save_obstacles(
        rows=50,
        cols=50,
        exclude_coords=[(49, 49), (0, 49), (49, 0)]
    )

if not os.path.exists("sensor_coords.txt"):
    setup_obstacles.generate_and_save_sensors(
        rows=50,
        cols=50,
        obstacle_file="obstacle_coords.txt",
        sensor_file="sensor_coords.txt",
        goal_coords=[(49, 49), (0, 49), (49, 0)],
        n_sensors=5
    )

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

##==============================================================
## GridWorldEnv Class
##==============================================================
class GridWorldEnv(Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_width = 50
        self.grid_height = 50
        self.max_steps = 500
        self.num_obstacles = 45

        self.action_space = Discrete(8)
        self.observation_space = Dict({
            "agent_pos": Box(low=0, high=max(self.grid_width, self.grid_height) - 1, shape=(2,), dtype=np.int32),
            "sensor_pos": Box(low=0, high=max(self.grid_width, self.grid_height) - 1, shape=(5, 2), dtype=np.int32),
            "battery_levels": Box(low=0.0, high=100.0, shape=(5,), dtype=np.float32),
            "exit_distances": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        })

        self.static_grid = np.full((self.grid_height, self.grid_width), '.', dtype='<U1')

        self.goal_positions = [(49, 49), (0, 49), (49, 0)]
        for gx, gy in self.goal_positions:
            self.static_grid[gx, gy] = 'G'

        self.fixed_obstacles = self.load_obstacles_from_file()
        self.original_sensor_batteries = self.load_sensors_with_batteries()
        self.sensor_batteries = dict(self.original_sensor_batteries)

        for (x, y) in self.fixed_obstacles:
            if (x, y) not in self.goal_positions:
                self.static_grid[x, y] = '#'

        for (x, y) in self.sensor_batteries:
            if self.static_grid[x, y] == '.':  
                self.static_grid[x, y] = 'S'

        pygame.init()
        self.fixed_window_size = 750
        self.cell_size = self.fixed_window_size // max(self.grid_width, self.grid_height)
        self.screen = pygame.display.set_mode((self.fixed_window_size, self.fixed_window_size))
        pygame.display.set_caption("GridWorld Visualization")
        self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))

        self.reset()

    def load_obstacles_from_file(self, filename="obstacle_coords.txt"):
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

    def load_sensors_with_batteries(self, filename="sensor_coords.txt"):
        sensors = {}
        try:
            with open(filename, "r") as f:
                for line in f:
                    r, c, battery = map(float, line.strip().split(","))
                    sensors[(int(r), int(c))] = battery
        except FileNotFoundError:
            print(f"[WARNING] Sensor file '{filename}' not found.")
        return sensors
    
    def get_sensor_distances(self, pos, normalize=True):
        return chebyshev_distances(pos, list(self.sensor_batteries.keys()), self.grid_width, self.grid_height, normalize)

    def get_exit_distances(self, normalize=True):
        return chebyshev_distances(self.agent_pos, self.goal_positions, self.grid_width, self.grid_height, normalize)
    
    def get_observation(self):
        agent_pos = self.agent_pos.copy()

        sorted_sensors = sorted(self.sensor_batteries.items())
        sensor_positions = np.array([list(pos) for pos, _ in sorted_sensors], dtype=np.int32)
        battery_levels = np.array([level for _, level in sorted_sensors], dtype=np.float32)

        while len(sensor_positions) < 5:
            sensor_positions = np.vstack((sensor_positions, [[-1, -1]]))
            battery_levels = np.append(battery_levels, -1.0)

        exit_distances = self.get_exit_distances(normalize=True)

        return {
            "agent_pos": agent_pos,
            "sensor_pos": sensor_positions[:5],
            "battery_levels": battery_levels[:5],
            "exit_distances": exit_distances
        }

    def reset(self, seed=None, options=None):
        self.grid = np.copy(self.static_grid)

        while True:
            x, y = random.randint(0, self.grid_height - 1), random.randint(0, self.grid_width - 1)
            if self.grid[x, y] == '.':
                self.agent_pos = np.array([x, y])
                break

        self.visited = {tuple(self.agent_pos)}
        self.sensor_batteries = dict(self.original_sensor_batteries)
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        return self.get_observation(), {}

    def f_distance(self):
        '''
        Reward based on distance from agent to closest exit
        '''
        exit_distances = get_exit_distances(normalize=False)
        d_min = min(exit_distances)
        norm = max(self.grid_height - 1, self.grid_width - 1)
        return np.exp(-d_min / norm)

    def f_wall(self):
        '''
        Reward that penalizes agent for colliding with walls
        '''
        n_collisions = 1
        n_steps = 1
        return np.exp(-n_collisions / n_steps)

    def f_battery(self):
        '''
        Reward that is based off the battery level of the nearest sensor
        (motivates agent to go along high-battery level paths)
        '''
        sensor_coords = list(self.sensor_batteries.keys())
        distances = chebyshev_distances(self.agent_pos, sensor_coords, self.grid_width, self.grid_height, normalize=False)
        nearest_index = int(np.argmin(distances))
        nearest_sensor = sensor_coords[nearest_index]
        battery_level = self.sensor_batteries.get(nearest_sensor, 0.0)
        return battery_level / 100

    def f_exit(self):
        '''
        Hard positive reward for when the agent reaches an exit
        (influenced by avg battery level along path travelled by agent
        to reach the exit)
        '''
        pass

    def step(self, action):
        self.episode_steps += 1

        depletion_rate = 0.01
        for coord in self.sensor_batteries:
            self.sensor_batteries[coord] = max(0.0, self.sensor_batteries[coord] - depletion_rate)

        old_dist = min(np.linalg.norm(self.agent_pos - np.array(goal)) for goal in self.goal_positions)
        reward = -1

        direction_map = {
            0: (-1,  0),
            1: (-1, +1),
            2: ( 0, +1),
            3: (+1, +1),
            4: (+1,  0),
            5: (+1, -1),
            6: ( 0, -1),
            7: (-1, -1),
        }
        move = direction_map[int(action)]
        new_pos = self.agent_pos + move

        if 0 <= new_pos[0] < self.grid_height and 0 <= new_pos[1] < self.grid_width:
            char = self.grid[tuple(new_pos)]
            if char == '#':
                reward = -5
                self.obstacle_hits += 1
            else:
                self.agent_pos = new_pos
                self.visited.add(tuple(self.agent_pos))
        else:
            reward = -10

        new_dist = min(np.linalg.norm(self.agent_pos - np.array(goal)) for goal in self.goal_positions)
        shaping_bonus = (old_dist - new_dist) * 2.0
        reward += shaping_bonus

        terminated = tuple(self.agent_pos) in self.goal_positions
        truncated = self.episode_steps >= self.max_steps

        if terminated:
            reward = 100

        self.total_reward += reward

        return self.get_observation(), reward, terminated, truncated, {
            "collisions": self.obstacle_hits,
            "steps": self.episode_steps,
            "agent_pos": self.agent_pos.copy()
        }

    def render_pygame(self):
        screen = self.screen
        font = self.font
        cell_size = self.cell_size

        colors = {
            '.': (255, 255, 255),
            '#': (100, 100, 100),
            '*': (255, 255, 0),
            'A': (0, 0, 255),
            'G': (0, 255, 0),
            'F': (0, 255, 255),
            'S': (255, 0, 0)
        }

        grid_copy = self.grid.copy()
        for pos in self.visited:
            x, y = pos
            if (x, y) != tuple(self.agent_pos) and (x, y) not in self.goal_positions:
                if self.static_grid[x, y] != 'S':  # don't overwrite sensor cells
                    grid_copy[x, y] = '*'


        ax, ay = self.agent_pos
        if tuple(self.agent_pos) in self.goal_positions:
            grid_copy[ax, ay] = 'F'
        else:
            grid_copy[ax, ay] = 'A'

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                val = grid_copy[i, j]
                color = colors.get(val, (0, 0, 0))
                pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))
                pygame.draw.rect(screen, (0, 0, 0), (j * cell_size, i * cell_size, cell_size, cell_size), 1)

                if (i, j) in self.sensor_batteries:
                    battery = self.sensor_batteries[(i, j)]
                    label = font.render(f"{int(battery)}", True, (0, 0, 0))
                    text_rect = label.get_rect(center=(j * cell_size + cell_size // 2, i * cell_size + cell_size // 2))
                    screen.blit(label, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        pygame.time.wait(100)

    def episode_summary(self):
        print(f"   Episode Summary:")
        print(f"   Total Steps     : {self.episode_steps}")
        print(f"   Obstacles Hit   : {self.obstacle_hits}")
        print(f"   Total Reward    : {self.total_reward}")
        print(f"   Sensor Battery Levels (Sample):")
        for i, ((x, y), battery) in enumerate(self.sensor_batteries.items()):
            print(f"     Sensor {i+1} at ({x},{y}) → {battery:.2f}")

    def close(self):
        pygame.quit()
