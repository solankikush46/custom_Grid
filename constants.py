# constants.py
import os

# grid symbols
EMPTY = '.'
OBSTACLE = '#'
GOAL = 'G'
SENSOR = 'S'
AGENT = 'A'
FINISHED = 'F'
TRAIL_OUTSIDE = '*'
TRAIL_INSIDE = 'T'
RADAR_BG = 'B'

# rgb colors for pygame rendering
RENDER_COLORS = {
    EMPTY: (255, 255, 255),
    OBSTACLE: (100, 100, 100),
    TRAIL_OUTSIDE: (255, 255, 0),
    AGENT: (0, 0, 255),
    GOAL: (0, 255, 0),
    FINISHED: (0, 255, 255),
    SENSOR: (255, 0, 0),
    TRAIL_INSIDE: (173, 216, 230),
    RADAR_BG: (255, 165, 0),
}

# actions agent can take (cardinal directions and diagonals)
DIRECTION_MAP = {
    0: (-1,  0), 1: (-1, +1), 2: (0, +1), 3: (+1, +1),
    4: (+1,  0), 5: (+1, -1), 6: (0, -1), 7: (-1, -1),
}

# model saving and logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "SavedModels")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODELS = {
    "ppo": os.path.join(MODEL_DIR, "PPO_custom_grid"),
    "dqn": os.path.join(MODEL_DIR, "DQN_custom_grid")
}

LOGS = {
    "ppo": os.path.join(LOG_DIR, "PPO_custom_grid"),
    "dqn": os.path.join(LOG_DIR, "DQN_custom_grid"),
}

# directory for storing generated ASCII grids
GRID_DIR = os.path.join(BASE_DIR, "saved_grids")
RANDOM_GRID_DIR = os.path.join(GRID_DIR, "random")
FIXED_GRID_DIR = os.path.join(GRID_DIR, "fixed")



