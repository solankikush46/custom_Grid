# reward_functions.py

import numpy as np
from utils import chebyshev_distances

##==============================================================
## Reward functions agent in GridWorldEnv may use
##==============================================================
# simple reward
#-----------------------
# reward agent for step closer to goal
def get_reward_a(env, new_pos):
    """
    Reward function encouraging:
    - Progress toward the goal
    - Avoiding invalid moves and revisits
    - Penalizing running low on battery
    - (No additional sensor reward beyond original code)
    """
    subrewards = {}

    # Reaching the goal: big positive reward, no penalties
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = env.n_rows * env.n_cols
        subrewards["time_penalty"] = 0
        subrewards["invalid_penalty"] = 0
        subrewards["revisit_penalty"] = 0
        subrewards["battery_penalty"] = 0
        subrewards["progress_shaping"] = 0
        return sum(subrewards.values()), subrewards

    # Time penalty (small penalty per step)
    subrewards["time_penalty"] = -0.05

    # Invalid move penalty
    if not env.can_move_to(new_pos):
        subrewards["invalid_penalty"] = -1.0
    else:
        subrewards["invalid_penalty"] = 0

    # Revisit penalty
    if new_pos in env.visited:
        subrewards["revisit_penalty"] = -0.25
    else:
        subrewards["revisit_penalty"] = 0

    # Battery penalty if critically low
    if env.current_battery_level <= 10:
        subrewards["battery_penalty"] = -50
    else:
        subrewards["battery_penalty"] = 0

    # Progress shaping (Chebyshev distance)
    prev_pos = env.agent_pos
    prev_dist = min(chebyshev_distances(prev_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=True))
    new_dist = min(chebyshev_distances(new_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=True))

    progress = prev_dist - new_dist
    subrewards["progress_shaping"] = 0.2 * progress

    reward = sum(subrewards.values())
    return reward, subrewards

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
