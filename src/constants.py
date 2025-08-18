# constants.py
import os

# ====================================================================
# --- Grid Symbol Definitions ---
# ===================================================================

# 1. Character symbols (for grid files and easy identification)
EMPTY_CHAR = '.'
OBSTACLE_CHAR = '#'
GOAL_CHAR = 'G'
SENSOR_CHAR = 'S'
GUIDED_MINER_CHAR = 'M'
BASE_STATION_CHAR = 'B'
MINER_CHAR = 'm'

# 2. Integer IDs (for efficient internal NumPy array representation)
EMPTY_ID = 0
OBSTACLE_ID = 1
SENSOR_ID = 2
BASE_STATION_ID = 3
GUIDED_MINER_ID = 4
GOAL_ID = 5
MINER_ID = 6

# 3. Mapping from characters to integer IDs (for loading grids)
CHAR_TO_INT_MAP = {
    EMPTY_CHAR: EMPTY_ID,
    OBSTACLE_CHAR: OBSTACLE_ID,
    GOAL_CHAR: GOAL_ID,
    SENSOR_CHAR: SENSOR_ID,
    GUIDED_MINER_CHAR: GUIDED_MINER_ID,
    BASE_STATION_CHAR: BASE_STATION_ID,
    MINER_CHAR: MINER_ID
}

# 4. RGB colors for pygame rendering (keys are the character symbols)
RENDER_COLORS = {
    EMPTY_CHAR: (255, 255, 255),
    OBSTACLE_CHAR: (100, 100, 100),
    GOAL_CHAR: (0, 255, 0),
    SENSOR_CHAR: (255, 0, 0),
    GUIDED_MINER_CHAR: (0, 0, 255),
    BASE_STATION_CHAR: (128, 0, 128),
    MINER_CHAR: (0, 100, 0),
    "TRAIL": (255, 255, 0),
    "DSTAR": (30, 144, 255),
}
DSTAR_PATH_THICKNESS = 4
TRAIL_PATH_THICKNESS = 4

# ===================================================================
# --- Other Constants ---
# ===================================================================
# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXED_GRID_DIR = os.path.join(BASE_DIR, "src", "saved_grids", "fixed")

# actions agent can take (cardinal directions and diagonals)
DIRECTION_MAP = {
    0: (-1,  0), 1: (-1, +1), 2: (0, +1), 3: (+1, +1),
    4: (+1,  0), 5: (+1, -1), 6: (0, -1), 7: (-1, -1),
}

ACTIONS = [DIRECTION_MAP[i] for i in range(len(DIRECTION_MAP))]

ACTION_NAMES = {
    0: "UP",
    1: "UP-RIGHT",
    2: "RIGHT",
    3: "DOWN-RIGHT",
    4: "DOWN",
    5: "DOWN-LEFT",
    6: "LEFT",
    7: "UP-LEFT"
}

MOVE_TO_ACTION_MAP = {move: action for action, move in DIRECTION_MAP.items()}

# model saving and logging
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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

SAVE_DIR = os.path.join(BASE_DIR, "saved_experiments")

# directory for storing generated ASCII grids
GRID_DIR = os.path.join(BASE_DIR, "saved_grids")
RANDOM_GRID_DIR = os.path.join(GRID_DIR, "random")
FIXED_GRID_DIR = os.path.join(GRID_DIR, "fixed")

##=================================
# Constants (for the Energy Model)
##=================================
BATTERY_CAPACITY_JOULES = 10.0 # Total Capacity of each battery
ALPHA_ELEC = 50e-9          # 50 nJ/bit
ALPHA_SHORT = 10e-12        # 10 pJ/bit/m^2
ALPHA_LONG = 0.0013e-12     # 0.0013 pJ/bit/m^4
THRESHOLD_DIST = 15.0  # Threshold distance in the grid
MAX_COMM_RANGE = 141 # 25.0           # Max Communication range (in grid cells)
# have to adjust other variables according to max_comm_range
#========================
# Message Sizes (in bits)
#========================
K_BROADCAST = 10_000 # Sensor-to-Sensor
K_TO_MINER = 5_000 # Miner-to-Sensor
K_TO_BASE = 2_000  # Sensor-to-base

##==============================================================
## Lookup table for use in SimulationController::sensor_cost_tier
##==============================================================
COST_TABLE = [0.0] * 101
for i in range(0,  6):   COST_TABLE[i] = 400.0
for i in range(6, 11):   COST_TABLE[i] = 200.0
for i in range(11, 21):  COST_TABLE[i] = 100.0
for i in range(21, 31):  COST_TABLE[i] =  50.0




