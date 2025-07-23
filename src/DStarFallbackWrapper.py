# DStarFallbackWrapper

import gym
import numpy as np
import torch as th # IMPORTANT: Add this import for torch operations
from src.constants import MOVE_TO_ACTION_MAP # Import the inverse map

class DStarFallbackWrapper(gym.Wrapper):
    """
    A wrapper that implements a hybrid control system with a D* Lite fallback.
    - If the agent's confidence (max action probability) is high, it uses the agent's action.
    - If confidence is low, it overrides the action with the "expert" action provided
      by the D* Lite path planner.
    """
    def __init__(self, env, model, confidence_threshold=0.75):
        super().__init__(env)
        if not hasattr(env, 'pathfinder'):
            raise AttributeError("The wrapped environment must have a 'pathfinder' attribute.")
        
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.last_obs = None

    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info

    def step(self, action_from_agent):
        """
        Intercepts the action from the PPO agent and decides whether to use it
        or fall back to the D* Lite planner.
        """
        expert_action = None # Initialize here for use in info dict

        # 1. Get action probabilities from the model's policy for the last known state.
        #    This is wrapped in 'no_grad' for efficiency.
        with th.no_grad():
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.last_obs)
            
            # --- START OF FIX ---
            # Use get_distribution() to get the policy's probability distribution
            distribution = self.model.policy.get_distribution(obs_tensor)
            # Get the probabilities from that distribution object
            probs = distribution.distribution.probs.cpu().numpy()[0]
            # --- END OF FIX ---

        max_prob = np.max(probs)
        
        # 2. The Fallback Logic
        if max_prob >= self.confidence_threshold:
            final_action = action_from_agent
        else:
            expert_action = self._get_dstar_action()
            final_action = expert_action if expert_action is not None else action_from_agent

        # 3. Take the determined step in the environment
        next_obs, reward, terminated, truncated, info = self.env.step(final_action)
        
        # 4. Store the latest observation and add debug info
        self.last_obs = next_obs
        info['confidence'] = float(max_prob)
        info['used_fallback'] = (expert_action is not None and final_action == expert_action)
        
        return next_obs, reward, terminated, truncated, info

    def _get_dstar_action(self):
        # This function does not need to be changed.
        path = self.env.pathfinder.get_path_to_goal()
        if not path or len(path) < 2:
            return None

        current_pos_rc = tuple(self.env.agent_pos)
        next_pos_xy = path[1]
        move_rc = (next_pos_xy[1] - current_pos_rc[0], next_pos_xy[0] - current_pos_rc[1])
        
        return MOVE_TO_ACTION_MAP.get(move_rc)
