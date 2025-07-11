# reward_functions.py

import numpy as np
from utils import *

##==============================================================
## Reward functions agent in GridWorldEnv may use
##==============================================================
# === Constants ===
LOWER_BOUND = -1
UPPER_BOUND = 10.0
BATTERY_THRESHOLD = 10
BASE_GOAL_REWARD = 100.0  # Reference reward for a 20x20 grid

# === Base subreward values ===
base_invalid_penalty = -1.0
base_battery_penalty = -25.0
base_revisit_penalty = -0.25
base_time_penalty = -0.05
min_progress_penalty = -1
base_progress_weight = 1.0

def get_reward_a(env, new_pos):
    subrewards = {}

    # === Grid-based scaling ===
    goal_reward = env.n_rows * env.n_cols
    scale = goal_reward / BASE_GOAL_REWARD

    # === Initialize subrewards ===
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = goal_reward
        subrewards["invalid_penalty"] = 0
        subrewards["battery_penalty"] = 0
        subrewards["progress_shaping"] = 0
        subrewards["revisit_penalty"] = 0
        subrewards["time_penalty"] = 0
    else:
        subrewards["goal_reward"] = 0
        subrewards["invalid_penalty"] = scale * (base_invalid_penalty if not env.can_move_to(new_pos) else 0)
        subrewards["battery_penalty"] = scale * (base_battery_penalty if env.current_battery_level <= BATTERY_THRESHOLD else 0)

        prev_pos = env.agent_pos
        prev_dist = min(chebyshev_distances(prev_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        new_dist = min(chebyshev_distances(new_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        progress = prev_dist - new_dist
        subrewards["progress_shaping"] = scale * base_progress_weight * progress

        subrewards["revisit_penalty"] = scale * (base_revisit_penalty if new_pos in env.visited else 0)
        subrewards["time_penalty"] = scale * base_time_penalty

    # === Compute raw and normalized reward ===
    raw_reward = sum(subrewards.values())
    min_reward = scale * (
        base_invalid_penalty +
        base_battery_penalty +
        base_revisit_penalty +
        base_time_penalty +
        min_progress_penalty
    )
    max_reward = goal_reward

    # normalize total reward and subrewards
    def normalize(val):
        norm = ((val - min_reward) / (max_reward - min_reward)) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND
        return np.clip(norm, LOWER_BOUND, UPPER_BOUND)

    norm_reward = normalize(raw_reward)
    #norm_subrewards = {k: normalize(v) for k, v in subrewards.items()}
    
    return norm_reward, subrewards

def get_reward_b(env, new_pos):
    subrewards = {}

    # Goal reward
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = 1.0
        subrewards["distance_penalty"] = 0
        subrewards["invalid_penalty"] = 0
        return subrewards["goal_reward"], subrewards

    # Obstacle penalty
    if not env.can_move_to(new_pos):
        subrewards["invalid_penalty"] = -1.0
        subrewards["distance_penalty"] = 0
        subrewards["goal_reward"] = 0
        return subrewards["invalid_penalty"], subrewards

    # Otherwise, distance penalty to closest goal
    min_dist = min(
        euclidean_distance(new_pos, goal_pos)
        for goal_pos in env.goal_positions
    )

    subrewards["distance_penalty"] = -0.35 * min_dist
    subrewards["goal_reward"] = 0
    subrewards["invalid_penalty"] = 0

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

# composite reward
#-----------------------
def f_distance(agent_pos, goal_positions, n_rows, n_cols):
    '''
    Reward based on distance from agent to closest exit
    '''
    distances = chebyshev_distances(agent_pos, goal_positions, n_cols, n_rows, normalize=False)
    d_min = min(distances)
    norm = max(n_rows - 1, n_cols - 1)
    return np.exp(-d_min / norm)


def f_wall(n_collisions, n_steps):
    '''
    Reward that penalizes agent for colliding with walls
    '''
    if n_steps == 0:
        return 1.0
    return np.exp(-n_collisions / n_steps)


def f_battery(agent_pos, sensor_batteries, n_cols, n_rows):
    '''
    Reward that is based off the battery level of the nearest sensor
    (motivates agent to go along high-battery level paths)
    '''
    sensor_coords = list(sensor_batteries.keys())
    if not sensor_coords:
        return 0.0

    distances = chebyshev_distances(agent_pos, sensor_coords, n_cols, n_rows, normalize=False)
    nearest_index = int(np.argmin(distances))
    nearest_sensor = sensor_coords[nearest_index]
    battery_level = sensor_batteries.get(nearest_sensor, 0.0)
    return battery_level / 100


def f_exit(agent_pos, goal_positions, battery_values_in_radar):
    '''
    Hard positive reward for when the agent reaches an exit
    (influenced by avg battery level along path travelled by agent
    to reach the exit)
    '''
    if tuple(agent_pos) in goal_positions:
        if battery_values_in_radar:
            average_battery = sum(battery_values_in_radar) / len(battery_values_in_radar)
            return average_battery
        else:
            return 0.0
    else:
        return 0.0

def compute_reward(env, new_pos):
    new_pos = tuple(new_pos)
    return get_reward_a(env, new_pos)
