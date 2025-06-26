# reward_functions.py

import numpy as np
from utils import chebyshev_distances

##==============================================================
## Reward functions agent in GridWorldEnv may use
##==============================================================
# simple reward
#-----------------------
def get_simple_reward(env, new_pos):
    reward = 0.0

    # if new position is exit?
    if (new_pos in env.goal_positions):
        reward = env.n_rows * env.n_cols # reward should make worst case scenario positive
        print("reached exit, reward =", reward)
    else:
        # penalty per timestep
        reward += -0.04

        # penalized for making invalid move
        if not env.can_move_to(new_pos):
            reward += -0.75

        # penalized for revisiting a cell
        if new_pos in env.visited:
            reward += -0.21

    return reward

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
    return get_simple_reward(env, new_pos)
