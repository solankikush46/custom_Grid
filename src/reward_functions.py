# reward_functions.py

import numpy as np
from src.utils import *

##==============================================================
## Reward functions agent in GridWorldEnv may use
##==============================================================
# === Constants ===
LOWER_BOUND = -1
UPPER_BOUND = 9.0
BATTERY_THRESHOLD = 10
BASE_GOAL_REWARD = 400.0  # Reference reward for a 20x20 grid

# === Base subreward values ===
base_invalid_penalty = -0.75
base_battery_penalty = -100
base_revisit_penalty = -0.25
base_time_penalty = -0.04
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
        #subrewards["progress_shaping"] = 0
        subrewards["revisit_penalty"] = 0
        subrewards["time_penalty"] = 0
    else:
        subrewards["goal_reward"] = 0
        subrewards["invalid_penalty"] = scale * (base_invalid_penalty if not env.can_move_to(new_pos) else 0)
        subrewards["battery_penalty"] = scale * (base_battery_penalty if env.current_battery_level <= BATTERY_THRESHOLD else 0)

        '''
        prev_pos = env.agent_pos
        prev_dist = min(chebyshev_distances(prev_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        new_dist = min(chebyshev_distances(new_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        progress = prev_dist - new_dist
        subrewards["progress_shaping"] = scale * base_progress_weight * progress
        '''

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
    b_inval_pen = -0.75
    b_rev_pen = -0.21
    b_t_pen = -0.04
    b_bat_pen = -100

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": env.n_rows * env.n_cols,
            "invalid_penalty": 0,
            "revisit_penalty": 0,
            "time_penalty": 0,
            "battery_penalty": 0
        }
    else:
        inval_pen = b_inval_pen if not env.can_move_to(new_pos) else 0
        rev_pen = b_rev_pen if new_pos in env.visited else 0
        t_pen = b_t_pen
        bat_pen = b_bat_pen if env.current_battery_level <= 10 else 0
        subrewards = {
            "goal_reward": 0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "time_penalty": t_pen,
            "battery_penalty": bat_pen
            }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_c(env, new_pos):
    '''
    smaller reward scale
    '''
    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": 1.0,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "time_penalty": 0.0,
            "battery_penalty": 0.0
        }
    else:
        inval_pen = -0.75 if not env.can_move_to(new_pos) else 0.0
        rev_pen = -0.25 if new_pos in env.visited else 0.0
        t_pen = -0.04
        bat_pen = -1.0 if env.current_battery_level <= 10 else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "time_penalty": t_pen,
            "battery_penalty": bat_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_d(env, new_pos):
    '''
    reward c but with distance penalty, and no time penalty
    '''
    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": 1.0,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        inval_pen = -0.75 if not env.can_move_to(new_pos) else 0.0
        rev_pen = -0.25 if new_pos in env.visited else 0.0
        bat_pen = -1.0 if env.current_battery_level <= 10 else 0.0
        dist_pen = -2.0 * env._compute_min_distance_to_goal() # product of normalized euclidean distance to closest goal

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 10
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e2(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 5 # was 10
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e3(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 2.5
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_6(env, old_pos):
    """
    A proactive, path-informed reward function (Version 2).

    This version guarantees that the returned 'subrewards' dictionary always
    contains the same set of keys, preventing KeyErrors during data logging.

    Args:
        env: The environment object, which must contain the pathfinder.
        old_pos (tuple): The agent's position *before* the action.
    """
    # --- Tunable Weights ---
    w_goal = 1.0
    w_invalid = -0.75
    w_revisit = -0.25
    w_path_progress = 1.0
    w_dangerous = -1.0 # Penalty for moving to a state from which the pathfinder, with its knowledge of costs, can no longer find a viable or reasonable path to the goal
    path_prog_norm = 201 # max battery penalty + cost of moving a square = 200 + 1

    subrewards = {
        "goal_reward": 0.0,
        "invalid_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_progress_reward": 0.0,
    }
    
    # Case 1: Goal is reached
    new_pos = tuple(env.agent_pos)
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if not env.can_move_to(new_pos):
        subrewards["invalid_penalty"] = w_invalid
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit
    
    cost_from_old_pos = env.get_path_cost(old_pos)
    cost_from_new_pos = env.get_path_cost(new_pos)
    
    if cost_from_new_pos == float('inf'):
        if cost_from_old_pos == float('inf'):
            path_reward = 0.0
        else:
            path_reward = w_dangerous
    else:
        if cost_from_old_pos == float('inf'):
             path_reward = -w_dangerous
        else:
            path_reward = w_path_progress * (cost_from_old_pos - cost_from_new_pos) / path_prog_norm # scaled to [-1, 1] # try larger scale, or lower scale of other things?

    subrewards["path_progress_reward"] = path_reward

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_7(env, old_pos):
    """
    reward_6 with time penalty and lower cost of batteries
    """
    # --- Tunable Weights ---
    w_goal = 1.0
    w_invalid = -0.75
    w_revisit = -0.25
    w_path_progress = 1.0
    w_dangerous = -1.0 # Penalty for moving to a state from which the pathfinder, with its knowledge of costs, can no longer find a viable or reasonable path to the goal
    path_prog_norm = 101
    w_time = -0.05

    subrewards = {
        "goal_reward": 0.0,
        "invalid_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_progress_reward": 0.0,
        "time_penalty": 0.0
    }
    
    # Case 1: Goal is reached
    new_pos = tuple(env.agent_pos)
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if not env.can_move_to(new_pos):
        subrewards["invalid_penalty"] = w_invalid
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit
    
    cost_from_old_pos = env.get_path_cost(old_pos)
    cost_from_new_pos = env.get_path_cost(new_pos)

    if cost_from_new_pos == float('inf'):
        if cost_from_old_pos == float('inf'):
            path_reward = 0.0
        else:
            path_reward = w_dangerous
    else:
        if cost_from_old_pos == float('inf'):
             path_reward = -w_dangerous
        else:
            path_reward = w_path_progress * (cost_from_old_pos - cost_from_new_pos) / path_prog_norm # scaled to [-1, 1] # try larger scale, or lower scale of other things?

    subrewards["path_progress_reward"] = path_reward

    subrewards["time_penalty"] = w_time

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_8(env, old_pos):
    """
    reward_6 with time penalty and lower cost of batteries
    """
    # --- Tunable Weights ---
    w_goal = 1.0
    w_invalid = -0.75
    w_revisit = -0.25
    w_path_progress = 1.0
    w_dangerous = -1.0 # Penalty for moving to a state from which the pathfinder, with its knowledge of costs, can no longer find a viable or reasonable path to the goal
    path_prog_norm = 101
    w_time = -0.05

    subrewards = {
        "goal_reward": 0.0,
        "invalid_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_progress_reward": 0.0,
        "time_penalty": 0.0
    }
    
    # Case 1: Goal is reached
    new_pos = tuple(env.agent_pos)
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if old_pos == new_pos:
        subrewards["invalid_penalty"] = w_invalid
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit
    
    cost_from_old_pos = env.get_path_cost(old_pos)
    cost_from_new_pos = env.get_path_cost(new_pos)

    if cost_from_new_pos == float('inf'):
        if cost_from_old_pos == float('inf'):
            path_reward = 0.0
        else:
            path_reward = w_dangerous
    else:
        if cost_from_old_pos == float('inf'):
             path_reward = -w_dangerous
        else:
            path_reward = w_path_progress * (cost_from_old_pos - cost_from_new_pos) # scaled to [-1, 1] # try larger scale, or lower scale of other things?

    subrewards["path_progress_reward"] = path_reward if subrewards["revisit_penalty"] == 0 else 0

    subrewards["time_penalty"] = w_time

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

'''
def get_reward_f(env, new_pos, w_battery = 10):
    """
    Returns (reward, cost, info)
    - reward: positive task progress (goal, distance)
    - cost: penalties for constraints (invalid moves, revisits, battery)
    - info: dictionary with individual components for debugging/logging
    """
    # Tunable weights
    w_goal = 1.0
    w_dist = 2.0
    w_invalid = 0.75
    w_revisit = 0.25
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    # Reward components
    if new_pos in env.goal_positions:
        reward = w_goal
        dist_pen = 0.0
    else:
        # Reward is negative distance to goal (encourages progress)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist
        reward = dist_pen  # distance penalty treated as negative reward here

    # Cost components (penalties for constraints)
    inval_pen = w_invalid if not env.can_move_to(new_pos) else 0.0
    rev_pen = w_revisit if new_pos in env.visited else 0.0
    b = env.current_battery_level
    bat_pen = w_battery * sigmoid((battery_threshold - b) / battery_threshold)

    cost = inval_pen + rev_pen + bat_pen

    info = {
        "goal_reward": w_goal if new_pos in env.goal_positions else 0.0,
        "distance_reward": dist_pen,
        "invalid_cost": inval_pen,
        "revisit_cost": rev_pen,
        "battery_cost": bat_pen
    }

    return reward, cost, info
'''

def compute_reward(env, old_pos, reward_fn):
    old_pos = tuple(old_pos)
    return reward_fn(env, old_pos)
