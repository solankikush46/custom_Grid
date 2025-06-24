# grid_env.py

from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random 
import os  
import pygame
import grid_gen
from constants import *

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

##==============================================================
## GridWorldEnv Class
##==============================================================
class GridWorldEnv(Env):
    def __init__(self, grid_height=50, grid_width=50,
                 n_obstacles=45, n_sensors=5,
                 obstacle_file=None, sensor_file=None):
        super(GridWorldEnv, self).__init__()

        ##=============== member variables ===============##
        # grid config
        self.n_rows = grid_height
        self.n_cols = grid_width
        self.max_steps = 10_000
        self.n_obstacles = n_obstacles
        self.n_sensors = n_sensors

        # pygame rendering
        self.pygame_initialized = False
        self.fixed_window_size = None
        self.cell_size = None
        self.screen = None
        self.font = None
        self.clock = None
        self.render_fps = 30

        # environment variables
        self.grid = None
        self.static_grid = np.full((self.n_rows, self.n_cols),
                                   EMPTY, dtype='<U1')
        self.agent_pos = None
        self.visited = None
        self.battery_values_in_radar = None
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        # files for fixed grids
        self.obstacle_file = obstacle_file
        self.sensor_file = sensor_file

        # observation/action spaces
        self._init_spaces()

        # static environment elements
        self._init_goals()
        self._init_obstacles()
        self._init_sensors()
        self._init_radar_zone()
        ##================================================##

        self.reset()

    def _init_spaces(self):
        self.action_space = Discrete(8)
        max_dim = max(self.n_cols, self.n_rows)
        self.observation_space = Dict({
            "agent_pos": Box(low=0, high=max_dim - 1, shape=(2,), dtype=np.int32),
            "sensor_pos": Box(low=0, high=max_dim - 1, shape=(5, 2), dtype=np.int32),
            "battery_levels": Box(low=0.0, high=100.0, shape=(5,), dtype=np.float32),
            "exit_distances": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        })

    def _init_goals(self):
        self.goal_positions = [
            (self.n_rows - 1, self.n_cols - 1),
            (0, self.n_cols - 1),
            (self.n_rows - 1, 0)
        ]
        for r, c in self.goal_positions:
            if self.in_bounds((r, c)) and self.is_empty((r, c)):
                self.static_grid[r, c] = GOAL

    def _get_goal_exclusion_zone(self):
        # goal is not placed in sensor radar range???
        return grid_gen.get_safe_zone_around(self.goal_positions,
                                             self.n_rows,
                                             self.n_cols,
                                             radius=2)

    def _init_obstacles(self):
        # get obstacle positions (either random or specified)
        if self.obstacle_file:
            self.fixed_obstacles = load_obstacles_from_file(self.obstacle_file)
        else:
            grid_gen.generate_and_save_obstacles(
                rows=self.n_rows,
                cols=self.n_cols,
                exclude_coords=self._get_goal_exclusion_zone(),
                filename="obstacle_coords.txt",
                n_obstacles=self.n_obstacles
            )
            self.fixed_obstacles = load_obstacles_from_file("obstacle_coords.txt")

        # place obstacles on grid
        for (r, c) in self.fixed_obstacles:
            if self.in_bounds((r, c)) and self.is_empty((r, c)):
                self.static_grid[r, c] = OBSTACLE

    def _init_sensors(self):
        # get sensor positions (either random or specified)
        if self.sensor_file:
            self.original_sensor_batteries = load_sensors_with_batteries(self.sensor_file)
        else:
            grid_gen.generate_and_save_sensors(
                rows=self.n_rows,
                cols=self.n_cols,
                obstacle_file="obstacle_coords.txt",
                sensor_file="sensor_coords.txt",
                goal_coords=self.goal_positions,
                n_sensors=self.n_sensors
            )
            self.original_sensor_batteries = load_sensors_with_batteries("sensor_coords.txt")

        # get sensor battery levels
        self.sensor_batteries = dict(self.original_sensor_batteries)

        # place sensors
        for (r, c) in self.sensor_batteries:
            if self.in_bounds((r, c)) and self.is_empty((r, c)):
                self.static_grid[r, c] = SENSOR

    def _init_radar_zone(self):
        self.sensor_radar_zone = grid_gen.compute_sensor_radar_zone(
            self.sensor_batteries.keys(),
            self.n_rows,
            self.n_cols
        )

    def is_empty(self, pos):
        return self.static_grid[pos[0], pos[1]] == EMPTY

    def _generate_random_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.n_obstacles:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.is_empty((r, c)):
                obstacles.add((r, c))
        return obstacles

    def _generate_random_sensors(self):
        sensors = {}
        while len(sensors) < self.n_sensors:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if (r, c) not in sensors and self.is_empty((r, c)):
                sensors[(r, c)] = 100.0
        return sensors

        
    def init_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.fixed_window_size = 750
            self.cell_size = self.fixed_window_size // max(self.n_cols, self.n_rows)
            self.screen = pygame.display.set_mode((self.fixed_window_size, self.fixed_window_size))
            pygame.display.set_caption("GridWorld Visualization")
            self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True

    def render_pygame(self, uncap_fps=False):
        if not self.pygame_initialized:
            self.init_pygame()

        screen = self.screen
        font = self.font
        cell_size = self.cell_size
        colors = RENDER_COLORS
        
        grid_copy = self.grid.copy()

        # Update trail cells
        for pos in self.visited:
            row, col = pos
            if (row, col) != tuple(self.agent_pos) and (row, col) not in self.goal_positions:
                if self.static_grid[row, col] != SENSOR:
                    if (row, col) in self.sensor_radar_zone:
                        grid_copy[row, col] = TRAIL_INSIDE  # trail in radar zone (pink)
                    else:
                        grid_copy[row, col] = TRAIL_OUTSIDE  # trail outside radar (yellow)

        # Update agent position
        row, col = self.agent_pos
        if tuple(self.agent_pos) in self.goal_positions:
            grid_copy[row, col] = FINISHED
        else:
            grid_copy[row, col] = AGENT

        # Draw each cell
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                val = grid_copy[row, col]

                # Always start by painting the cell white
                pygame.draw.rect(screen, colors[EMPTY], (col * cell_size, row * cell_size, cell_size, cell_size))

                # Then radar zone (orange) if applicable (and not obstacle)
                if (row, col) in self.sensor_radar_zone and self.static_grid[row, col] != '#':
                    pygame.draw.rect(screen, colors[RADAR_BG], (col * cell_size, row * cell_size, cell_size, cell_size))

                # Then foreground item (trail, agent, goal, etc.)
                if val != EMPTY:
                    color = colors.get(val, (0, 0, 0))
                    pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

                # Border
                pygame.draw.rect(screen, (0, 0, 0), (col * cell_size, row * cell_size, cell_size, cell_size), 1)

        # Draw sensors and battery labels (on top of all)
        for (row, col), battery in self.sensor_batteries.items():
            pygame.draw.rect(screen, colors[SENSOR], (col * cell_size, row * cell_size, cell_size, cell_size))
            label = font.render(f"{int(battery)}", True, (0, 0, 0))
            text_rect = label.get_rect(center=(col * cell_size + cell_size // 2, row * cell_size + cell_size // 2))
            screen.blit(label, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    exit()

        if uncap_fps:
            self.clock.tick(15)
        else:
            self.clock.tick(self.render_fps) # use default render_fps
       
    def get_sensor_distances(self, pos, normalize=True):
        return chebyshev_distances(pos, list(self.sensor_batteries.keys()), self.n_cols, self.n_rows, normalize)

    def get_exit_distances(self, normalize=True):
        return chebyshev_distances(self.agent_pos, self.goal_positions, self.n_cols, self.n_rows, normalize)
    
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

        self.battery_values_in_radar = []

        while True:
            x, y = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.grid[x, y] == EMPTY:
                self.agent_pos = np.array([x, y])
                break

        self.visited = {tuple(self.agent_pos)}
        self.sensor_batteries = dict(self.original_sensor_batteries)
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        self.sensor_radar_zone = grid_gen.compute_sensor_radar_zone(self.sensor_batteries.keys(), self.n_rows, self.n_cols)

        return self.get_observation(), {}

    def can_move_to(self, pos):
        '''
        Return True if the position is in bounds
        and not blocked by an obstacle or sensor
        (sensors are placed on obstacles)
        '''
        return self.in_bounds(pos) and \
            self.static_grid[pos[0], pos[1]] != OBSTACLE and \
                self.static_grid[pos[0], pos[1]] != SENSOR

    def f_distance(self):
        '''
        Reward based on distance from agent to closest exit
        '''
        exit_distances = self.get_exit_distances(normalize=False)
        d_min = min(exit_distances)
        norm = max(self.n_rows - 1, self.n_cols - 1)
        return np.exp(-d_min / norm)

    def f_wall(self):
        '''
        Reward that penalizes agent for colliding with walls
        '''
        n_collisions = self.obstacle_hits
        n_steps = self.episode_steps
        return np.exp(-n_collisions / n_steps)

    def f_battery(self):
        '''
        Reward that is based off the battery level of the nearest sensor
        (motivates agent to go along high-battery level paths)
        '''
        sensor_coords = list(self.sensor_batteries.keys())
        distances = chebyshev_distances(self.agent_pos, sensor_coords, self.n_cols, self.n_rows, normalize=False)
        nearest_index = int(np.argmin(distances))
        nearest_sensor = sensor_coords[nearest_index]
        battery_level = self.sensor_batteries.get(nearest_sensor, 0.0)
        return battery_level / 100

    def f_time(self):
        k = 5
        return np.exp(-k * self.episode_steps)

    def f_exit(self):
        '''
        Hard positive reward for when the agent reaches an exit,
        influenced by the average battery level along the path travelled by the agent.
        '''
        if tuple(self.agent_pos) in self.goal_positions:
            if self.battery_values_in_radar:
                average_battery = sum(self.battery_values_in_radar) / len(self.battery_values_in_radar)
                return average_battery
            else:
                return 0.0
        else:
            return 0.0

    def in_bounds(self, pos):
        return 0 <= pos[0] < self.n_rows \
    and 0 <= pos[1] < self.n_cols

    def hit_wall(self, pos):
        return self.grid[tuple(pos)] == OBSTACLE

    def step(self, action):
        move = DIRECTION_MAP[int(action)]
        new_pos = self.agent_pos + move

        self.episode_steps += 1

        # deplete sensor battery levels
        for coord in self.sensor_batteries:
            self.sensor_batteries[coord] = max(0.0, self.sensor_batteries[coord] - 0.01)

        # check if in range of sensor
        for sensor_pos, battery in self.sensor_batteries.items():
            if self._in_radar(sensor_pos, self.agent_pos, radius=2):
                self.battery_values_in_radar.append(battery)

        # check for collisions with walls
        if self.can_move_to(new_pos):
            self.agent_pos = new_pos
            self.visited.add(tuple(self.agent_pos))
            hit_wall = False
        else:
            self.obstacle_hits += 1
            hit_wall = True

        terminated = tuple(self.agent_pos) in self.goal_positions
        truncated = self.episode_steps >= self.max_steps

        '''
        reward = (
            0.3 * self.f_distance()
            - 1.0 * float(hit_wall)
            + 0.3 * self.f_battery()
            + 0.3 * self.f_exit()
            - 0.01 * self.f_time()
        )
        '''
      
        # decay = np.exp(-5 * self.episode_steps / self.max_steps)
        reward = (
            0.2 * self.f_distance()
            #- float(self.hit_wall(new_pos))
            + 0.2 * self.f_wall()
            + 0.3 * self.f_battery()
            + 0.3 * self.f_exit()
            #- 0.01 * self.f_time()
        )
        s = "action: %s, f_dist: %s, f_wall: %s, f_battery: %s , f_exit: %s"
        '''
        print (s % (action, 0.2 * self.f_distance(),
                    float(self.hit_wall(new_pos)),
                    0.3 * self.f_battery(), 0.3 * self.f_exit()))
        '''
        self.total_reward += reward

        return self.get_observation(), reward, terminated, truncated, {
            "collisions": self.obstacle_hits,
            "steps": self.episode_steps,
            "agent_pos": self.agent_pos.copy()
        }
    
    
    def episode_summary(self):
        print(f"   Episode Summary:")
        print(f"   Total Steps     : {self.episode_steps}")
        print(f"   Obstacles Hit   : {self.obstacle_hits}")
        print(f"   Total Reward    : {self.total_reward}")
        print(f"   Sensor Battery Levels (Sample):")
        for i, ((x, y), battery) in enumerate(self.sensor_batteries.items()):
            print(f"     Sensor {i+1} at ({x},{y}) → {battery:.2f}")
        if self.battery_values_in_radar:
            avg_battery = sum(self.battery_values_in_radar) / len(self.battery_values_in_radar)
            print(f"   Avg Battery Level While in Sensor Radar: {avg_battery:.2f}")
        else:
            print("   Agent never entered a sensor radar zone.")

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
        
    def _in_radar(self, sensor_pos, agent_pos, radius=2):
        dx = abs(sensor_pos[0] - agent_pos[0])
        dy = abs(sensor_pos[1] - agent_pos[1])
        return dx <= radius and dy <= radius

    def manual_control_loop(self):
        self.init_pygame()
        obs, _ = self.reset()
        done = False

        key_to_action = {
            # 8 directional WASDQEZC
            pygame.K_w: 0,  # North
            pygame.K_e: 1,  # North-East
            pygame.K_d: 2,  # East
            pygame.K_c: 3,  # South-East
            pygame.K_s: 4,  # South
            pygame.K_z: 5,  # South-West
            pygame.K_a: 6,  # West
            pygame.K_q: 7,  # North-West

            # arrow keys (cardinal directions only)
            pygame.K_UP:    0,  # N
            pygame.K_RIGHT: 2,  # E
            pygame.K_DOWN:  4,  # S
            pygame.K_LEFT:  6,  # W
        }

        last_move_time = 0
        move_delay = 80 # milliseconds between moves

        while not done:
            self.render_pygame(uncap_fps=True)

            current_time = pygame.time.get_ticks()
        
            # Event handling (quit + escape)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close()
                    return

            # Handle held-down keys
            if current_time - last_move_time >= move_delay:
                keys = pygame.key.get_pressed()
                for key, action in key_to_action.items():
                    if keys[key]:
                        obs, reward, terminated, truncated, info = self.step(action)
                        print(f"Action: {action}, Reward: {reward:.3f}, Pos: {info['agent_pos']}")
                        done = terminated or truncated
                        last_move_time = current_time
                        break # only process one direction at a time

                
