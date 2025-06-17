from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pygame

class GridWorldEnv(Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 20
        self.max_steps = 200
        self.num_obstacles = 30

        self.action_space = Discrete(8)
        self.observation_space = Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        # Fixed grid setup
        self.static_grid = np.full((self.grid_size, self.grid_size), '.', dtype='<U1')

        # Define multiple goals
        self.goal_positions = [(19, 19), (0, 19), (19, 0)]
        for gx, gy in self.goal_positions:
            self.static_grid[gx, gy] = 'G'


        # Fixed obstacle positions
        self.fixed_obstacles = [
            (5, 5), (5, 6), (5, 7), (6, 5), (7, 5),
            (10, 10), (10, 11), (11, 10), (9, 10), (10, 9),
            (15, 15), (15, 16), (16, 15), (14, 15), (15, 14),
            (3, 12), (4, 12), (5, 12), (6, 12), (7, 12),
            (12, 3), (12, 4), (12, 5), (12, 6), (12, 7),
            (0, 10), (19, 10), (10, 0), (10, 19), (10, 15)
        ]

        for (x, y) in self.fixed_obstacles:
            if (x, y) not in self.goal_positions:
                self.static_grid[x, y] = '#'


        self.reset()

    def reset(self, seed=None, options=None):
        self.grid = np.copy(self.static_grid)

        # Randomize agent starting position, avoiding obstacles and goals
        while True:
            x, y = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if self.grid[x, y] == '.':
                self.agent_pos = np.array([x, y])
                break

        self.visited = {tuple(self.agent_pos)}

        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0

        return np.array(self.agent_pos, dtype=np.int32), {}

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

        return np.array(self.agent_pos, dtype=np.int32), reward, terminated, truncated, {
        "collisions": self.obstacle_hits,
        "steps": self.episode_steps,
        "agent_pos": self.agent_pos.copy()
        }


    def _min_dist_to_goal(self, pos):
        return min(np.linalg.norm(pos - np.array(goal)) for goal in self.goal_positions)
        
    def render(self):
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

        print("\n".join(" ".join(cell for cell in row) for row in grid_copy))
        print()

    def render_pygame(self, cell_size=30):
        pygame.init()
        window_size = self.grid_size * cell_size
        screen = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("GridWorld Visualization")

        colors = {
            '.': (255, 255, 255),
            '#': (100, 100, 100),
            '*': (255, 255, 0),
            'A': (0, 0, 255),
            'G': (0, 255, 0),
            'F': (0, 255, 255)
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

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        pygame.time.wait(100)

    def episode_summary(self):
        print(f"   Episode Summary:")
        print(f"   Total Steps     : {self.episode_steps}")
        print(f"   Obstacles Hit   : {self.obstacle_hits}")
        print(f"   Total Reward    : {self.total_reward}")
        print()

    def close(self):
        pygame.quit()
