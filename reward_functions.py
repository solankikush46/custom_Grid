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
        dist_pen = -2.0 * env._compute_min_distance_to_goal() # product of normalized euclid distance to closest goal

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

def compute_reward(env, new_pos):
    new_pos = tuple(new_pos)
    return get_reward_e(env, new_pos)
