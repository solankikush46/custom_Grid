# grid_env.py

from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import random 
import os  
import pygame
import grid_gen
from constants import *
from utils import *
from reward_functions import compute_reward

##==============================================================
## GridWorldEnv Class
##==============================================================
class GridWorldEnv(Env):
    def __init__(self,
                 grid_file: str = None,
                 grid_height: int = None,
                 grid_width: int = None,
                 obstacle_percentage=None,
                 n_sensors=None
                 ):
        super(GridWorldEnv, self).__init__()

        ##=============== member variables ===============##
        # grid config
        self._init_grid_config(grid_file,
                               grid_height, grid_width,
                               obstacle_percentage,
                               n_sensors)
        self.max_steps = 1_000

        # pygame rendering
        self.pygame_initialized = False
        self.fixed_window_size = None
        self.cell_size = None
        self.screen = None
        self.font = None
        self.clock = None
        self.render_fps = 160

        # environment state variables
        self.grid = self.static_grid.copy()
        self.agent_pos = None
        self.visited = None
        self.battery_values_in_radar = None
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0
        self.last_action = -1
            
        # observation/action spaces
        self._init_spaces()

        # radar around sensors
        self._init_radar_zone()
        ##================================================##

        self.reset()

    def _init_grid_config(self, grid_file, grid_height, grid_width, obstacle_percentage, n_sensors):
        if grid_file:
            grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
            self.static_grid, self.agent_pos, self.goal_positions, self.sensor_batteries = grid_gen.load_grid(grid_path)
            self.n_rows, self.n_cols = self.static_grid.shape
        else:
            assert grid_height is not None and grid_width is not None, \
                "Must provide grid_height and grid_width when not using a grid_file."
            self.n_rows = grid_height
            self.n_cols = grid_width
            save_path = os.path.join(RANDOM_GRID_DIR, f"grid_{self.n_rows}x{self.n_cols}{obstacle_percentage*100}p.txt")
            
            self.static_grid, self.agent_pos, self.goal_positions, self.sensor_batteries = grid_gen.gen_and_save_grid(
                self.n_rows, self.n_cols,
                obstacle_percentage=obstacle_percentage,
                n_sensors=n_sensors,
                place_agent=False,
                save_path=save_path
            )
        self.original_sensor_batteries = self.sensor_batteries

    def _init_spaces(self):
        self.action_space = Discrete(8)
        max_dim = max(self.n_cols, self.n_rows)
        '''
        self.observation_space = Dict({
            "agent_pos": Box(low=0, high=max_dim - 1, shape=(2,), dtype=np.int32),
            "sensor_pos": Box(low=0, high=max_dim - 1, shape=(self.n_sensors, 2), dtype=np.int32),
            "battery_levels": Box(low=0.0, high=100.0, shape=(self.n_sensors,), dtype=np.float32),
            "exit_distances": Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        })
        '''
        '''
        self.observation_space = Box(low=0, high=max_dim - 1,
                                     shape=(2,), dtype=np.int32)
        '''
        '''
        self.observation_space = Box(low=0, high=8,
                                     shape=(10, ), dtype=np.int32)
        # 10 comes from 3x3 view agent has, plus 1 for last action
        '''
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32
        )
        
    def _get_goal_exclusion_zone(self):
        # goal is not placed in sensor radar range???
        return grid_gen.get_safe_zone_around(self.goal_positions,
                                             self.n_rows,
                                             self.n_cols,
                                             radius=2)

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
                if self.can_move_to((row, col)):
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

    """
    def get_observation(self):
        r, c = self.agent_pos

        # pad grid with OBSTACLE so agent can see walls representing
        # "out of bounds"
        padded = np.full((self.n_rows + 2, self.n_cols + 2), OBSTACLE, dtype='<U1')
        padded[1:-1, 1:-1] = self.grid

        r_p, c_p = r + 1, c + 1
        local_view = padded[r_p - 1:r_p + 2, c_p - 1:c_p + 2]

        # 1 if obstacle or sensor, 0 otherwise
        is_blocked = np.isin(local_view, [OBSTACLE, SENSOR]).astype(np.int32)

        # flatten to 1D array
        flat_view = is_blocked.flatten()

        # add last action
        obs = np.concatenate([flat_view, [self.last_action]])
    
        return obs
    
        '''
        agent_pos = self.agent_pos.copy()

        sorted_sensors = sorted(self.sensor_batteries.items())
        sensor_positions = np.array([list(pos) for pos, _ in sorted_sensors], dtype=np.int32)
        battery_levels = np.array([level for _, level in sorted_sensors], dtype=np.float32)

        # for case where zero sensors
        if sorted_sensors:
            sensor_positions = np.array([list(pos) for pos, _ in sorted_sensors], dtype=np.int32)
            battery_levels = np.array([level for _, level in sorted_sensors], dtype=np.float32)
        else:
            sensor_positions = np.empty((0, 2), dtype=np.int32)
            battery_levels = np.empty((0,), dtype=np.float32)
        
        while len(sensor_positions) < self.n_sensors:
            sensor_positions = np.vstack((sensor_positions, [[-1, -1]]))
            battery_levels = np.append(battery_levels, -1.0)

        exit_distances = self.get_exit_distances(normalize=True)

        return {
            "agent_pos": agent_pos,
            "sensor_pos": sensor_positions[:self.n_sensors],
            "battery_levels": battery_levels[:self.n_sensors],
            "exit_distances": exit_distances
        }
        '''
    """
    def get_observation(self):
        r, c = self.agent_pos

        # 3x3 local view, padded with obstacles
        padded = np.full((self.n_rows + 2, self.n_cols + 2), OBSTACLE, dtype='<U1')
        padded[1:-1, 1:-1] = self.grid

        r_p, c_p = r + 1, c + 1
        local_view = padded[r_p - 1:r_p + 2, c_p - 1:c_p + 2]

        # 1 if obstacle or sensor, 0 otherwise
        is_blocked = np.isin(local_view, [OBSTACLE, SENSOR]).astype(np.float32)
        flat_view = is_blocked.flatten()  # shape (9,)

        # Normalize position to [0,1]
        norm_row = r / (self.n_rows - 1)
        norm_col = c / (self.n_cols - 1)

        # Chebyshev distance to nearest goal, normalized
        distances = chebyshev_distances(
            self.agent_pos,
            self.goal_positions,
            self.n_cols,
            self.n_rows,
            normalize=True
        )
        min_dist = min(distances)

        # Normalize last action (0–7) to [0,1]
        norm_last_action = self.last_action / 7.0

        # Concatenate all parts
        obs = np.concatenate([
            flat_view,               # (9,)
            [norm_row, norm_col],    # (2,)
            [min_dist],              # (1,)
            [norm_last_action]       # (1,)
        ])

        return obs

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the starting state for a new episode.
        Returns the initial observation and an empty info dict.
        """
        # register seed
        super().reset(seed=seed)

        # restore static grid layout
        self.grid = np.copy(self.static_grid)

        # reset state trackers
        self.battery_values_in_radar = []
        self.sensor_batteries = dict(self.original_sensor_batteries)
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        # place agent
        while True:
            x, y = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.grid[x, y] == EMPTY:
                self.agent_pos = np.array([x, y])
                break

        self.visited = {tuple(self.agent_pos)}
        

        # recompute radar zone
        '''
        self.sensor_radar_zone = grid_gen.compute_sensor_radar_zone(self.sensor_batteries.keys(), self.n_rows, self.n_cols)
        '''

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

    def in_bounds(self, pos):
        return 0 <= pos[0] < self.n_rows \
    and 0 <= pos[1] < self.n_cols

    def hit_wall(self, pos):
        return self.grid[tuple(pos)] == OBSTACLE or \
            self.grid[tuple(pos)] == SENSOR

    def agent_reached_exit(self):
        return tuple(self.agent_pos) in self.goal_positions
    
    def step(self, action):
        move = DIRECTION_MAP[int(action)]
        new_pos = self.agent_pos + move
        self.episode_steps += 1
        
        # compute reward, then mark new_pos as visited
        reward = compute_reward(self, new_pos)
        self.total_reward += reward

        self.visited.add(tuple(new_pos))       

        # move agent
        hit_wall = not self.can_move_to(new_pos)
        if not hit_wall:
            self.agent_pos = new_pos
        else:
            self.obstacle_hits += 1

        # deplete sensor battery values over time
        for coord in self.sensor_batteries:
            self.sensor_batteries[coord] = max(0.0, self.sensor_batteries[coord] - 0.01)

        # add battery values encountered in path to list
        for sensor_pos, battery in self.sensor_batteries.items():
            if self._in_radar(sensor_pos, self.agent_pos, radius=2):
                self.battery_values_in_radar.append(battery)

        '''
        s = "action: %s, f_dist: %s, f_wall: %s, f_battery: %s , f_exit: %s"
        print (s % (action, 0.2 * self.f_distance(),
                    float(self.hit_wall(new_pos)),
                    0.3 * self.f_battery(), 0.3 * self.f_exit()))
        '''

        terminated = self.agent_reached_exit()
        truncated = self.episode_steps >= self.max_steps

        ret = self.get_observation(), reward, terminated, truncated, {
            "collisions": self.obstacle_hits,
            "steps": self.episode_steps,
            "agent_pos": self.agent_pos.copy()
        }

        self.last_action = action
        return ret
    
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

        self.close()  
