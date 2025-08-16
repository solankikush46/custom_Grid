# reward_functions.py

def compute_reward(env, reward_fn, **kwargs):
    """
    Passes env and any per-step kwargs into the reward.
    Returns (reward: float, subrewards: dict).
    """
    return reward_fn(env, **kwargs)

def reward_d(env, new_pos):
    """
    Reward D (original semantics), adapted to the new env via small adapters:
      - env.goal_positions  -> MineEnv.goal_positions (property)
      - env.can_move_to(p)  -> MineEnv.can_move_to (simulator check)
      - env.visited         -> MineEnv.visited (property, pre-mutation)
      - env.current_battery_level -> set by MineEnv.step() for current cell
      - env._compute_min_distance_to_goal() -> normalized Euclidean distance
    """
    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": 1.0,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0,
        }
    else:
        inval_pen = -0.75 if not env.can_move_to(new_pos) else 0.0
        rev_pen   = -0.25 if new_pos in env.visited else 0.0
        bat_pen   = -1.0 if (getattr(env, "current_battery_level", None) is not None
                             and env.current_battery_level <= 10.0) else 0.0
        dist_pen  = -2.0 * float(env._compute_min_distance_to_goal())

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen,
        }

    total_reward = float(sum(subrewards.values()))
    return total_reward, subrewards
