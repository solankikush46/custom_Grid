# reward_functions.py

def compute_reward(env, reward_fn, **kwargs):
    """
    Passes env and any per-step kwargs into the reward.
    Returns (reward: float, subrewards: dict).
    """
    return reward_fn(env, **kwargs)

def reward_d(env, new_pos):
    """
    Reward D (original semantics), assuming MineEnv provides:
      - env.get_goal_positions() -> iterable of (r, c)
      - env.get_visited()        -> iterable/set of (r, c)
      - env.can_move_to(p)       -> bool
      - env.current_battery_level: float or None
      - env._compute_min_distance_to_goal() -> float in [0, 1]
    """
    goals = set(env.get_goal_positions())
    visited = set(env.get_visited())

    at_goal = (new_pos in goals)
    can_move = bool(env.can_move_to(new_pos))
    dist_pen = -2.0 * float(env._compute_min_distance_to_goal())

    bat_level = env.current_battery_level
    bat_pen = -1.0 if (bat_level is not None and bat_level <= 10.0) else 0.0

    inval_pen = 0.0 if can_move else -0.75
    rev_pen   = -0.25 if (new_pos in visited) else 0.0

    if at_goal:
        subrewards = {
            "goal_reward":      1.0,
            "invalid_penalty":  0.0,
            "revisit_penalty":  0.0,
            "battery_penalty":  0.0,
            "distance_penalty": 0.0,
        }
    else:
        subrewards = {
            "goal_reward":      0.0,
            "invalid_penalty":  inval_pen,
            "revisit_penalty":  rev_pen,
            "battery_penalty":  bat_pen,
            "distance_penalty": dist_pen,
        }

    total_reward = float(sum(subrewards.values()))
    return total_reward, subrewards
