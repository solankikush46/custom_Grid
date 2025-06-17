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

class GridWorldEnv(Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 50
        self.max_steps = 500
        self.num_obstacles = 45

        self.action_space = Discrete(8)
        self.observation_space = Dict({
            "agent_pos": Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32),
            "sensor_pos": Box(low=0, high=self.grid_size - 1, shape=(5, 2), dtype=np.int32),
            "battery_levels": Box(low=0.0, high=100.0, shape=(5,), dtype=np.float32),
            "exit_distances": Box(low=0.0, high=np.sqrt(2) * self.grid_size, shape=(3,), dtype=np.float32),
        })

        # Fixed grid setup
        self.static_grid = np.full((self.grid_size, self.grid_size), '.', dtype='<U1')

        # Define multiple goals
        self.goal_positions = [(49, 49), (0, 49), (49, 0)]
        for gx, gy in self.goal_positions:
            self.static_grid[gx, gy] = 'G'

        # Fixed obstacle positions
        self.fixed_obstacles = self.load_obstacles_from_file()
        self.original_sensor_batteries = self.load_sensors_with_batteries()
        self.sensor_batteries = dict(self.original_sensor_batteries)

        for (x, y) in self.fixed_obstacles:
            if (x, y) not in self.goal_positions:
                self.static_grid[x, y] = '#'
                
        for (x, y) in self.sensor_batteries:
            if self.static_grid[x, y] == '.':  
                self.static_grid[x, y] = 'S'

        # Pygame setup (rendering window initialized once)
        pygame.init()
        self.fixed_window_size = 800
        self.cell_size = self.fixed_window_size // self.grid_size
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
    
    def get_observation(self):
        agent_pos = self.agent_pos.copy()
        
        # Sensor positions and batteries (sorted for consistency)
        sorted_sensors = sorted(self.sensor_batteries.items())
        sensor_positions = np.array([list(pos) for pos, _ in sorted_sensors], dtype=np.int32)
        battery_levels = np.array([level for _, level in sorted_sensors], dtype=np.float32)

        # Pad in case < 5 sensors (safety for future-proofing)
        while len(sensor_positions) < 5:
            sensor_positions = np.vstack((sensor_positions, [[-1, -1]]))
            battery_levels = np.append(battery_levels, -1.0)

        # Distance from agent to each goal
        exit_distances = np.array([
            np.linalg.norm(agent_pos - np.array(goal)) for goal in self.goal_positions
        ], dtype=np.float32)

        return {
            "agent_pos": agent_pos,
            "sensor_pos": sensor_positions[:5],
            "battery_levels": battery_levels[:5],
            "exit_distances": exit_distances
        }

 

    def reset(self, seed=None, options=None):
        self.grid = np.copy(self.static_grid)

        # Randomize agent starting position, avoiding obstacles and goals
        while True:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[x, y] == '.':
                self.agent_pos = np.array([x, y])
                break

        self.visited = {tuple(self.agent_pos)}
        self.sensor_batteries = dict(self.original_sensor_batteries)
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        return self.get_observation(), {}

    def step(self, action):
        direction_map = {
            0: (-1,  0),  # N
            1: (-1, +1),  # NE
            2: ( 0, +1),  # E
            3: (+1, +1),  # SE
            4: (+1,  0),  # S
            5: (+1, -1),  # SW
            6: ( 0, -1),  # W
            7: (-1, -1),  # NW
        }

        move = direction_map[int(action)]
        new_pos = self.agent_pos + move

        self.episode_steps += 1
        reward = -1  # base penalty per step

        # Deplete sensor battery levels
        depletion_rate = 0.01
        for coord in self.sensor_batteries:
            self.sensor_batteries[coord] = max(0.0, self.sensor_batteries[coord] - depletion_rate)

        # Calculate old distance to nearest goal
        old_dist = min(np.linalg.norm(self.agent_pos - np.array(goal)) for goal in self.goal_positions)

        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            char = self.grid[tuple(new_pos)]
            if char == '#':
                reward = -5
                self.obstacle_hits += 1
            else:
                self.agent_pos = new_pos
                self.visited.add(tuple(self.agent_pos))
        else:
            reward = -10  # out of bounds

        # New distance to nearest goal
        new_dist = min(np.linalg.norm(self.agent_pos - np.array(goal)) for goal in self.goal_positions)
        shaping_bonus = (old_dist - new_dist) * 2.0
        reward += shaping_bonus

        terminated = tuple(self.agent_pos) in self.goal_positions
        truncated = self.episode_steps >= self.max_steps

        if terminated:
            reward = 100  # big reward for reaching any goal

        self.total_reward += reward

        return self.get_observation(), reward, terminated, truncated, {
        "collisions": self.obstacle_hits,
        "steps": self.episode_steps,
        "agent_pos": self.agent_pos.copy()
        }


    def _min_dist_to_goal(self, pos):
        return min(np.linalg.norm(pos - np.array(goal)) for goal in self.goal_positions)

    def render_pygame(self):
        screen = self.screen
        font = self.font
        cell_size = self.cell_size

        colors = {
            '.': (255, 255, 255),  # Empty
            '#': (100, 100, 100),  # Obstacle
            '*': (255, 255, 0),    # Visited
            'A': (0, 0, 255),      # Agent
            'G': (0, 255, 0),      # Goal
            'F': (0, 255, 255),    # Finished at goal
            'S': (255, 0, 0)       # Sensor
        }

        grid_copy = self.grid.copy()
        for pos in self.visited:
            x, y = pos
            if (x, y) != tuple(self.agent_pos) and (x, y) not in self.goal_positions:
                grid_copy[x, y] = '*'

        ax, ay = self.agent_pos
        if tuple(self.agent_pos) in self.goal_positions:
            grid_copy[ax, ay] = 'F'
        else:
            grid_copy[ax, ay] = 'A'

        for i in range(self.grid_size):
            for j in range(self.grid_size):
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
