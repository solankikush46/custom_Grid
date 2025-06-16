import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []
        self.episode_count = 0

    def _on_training_start(self) -> None:
        self.current_rewards = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            self.current_rewards[i] += rewards[i]

            if done:
                ep_reward = self.current_rewards[i]
                info = infos[i]

                # Extract custom info
                collisions = info.get("collisions", -1)
                steps = info.get("steps", -1)
                agent_pos = np.array(info.get("agent_pos", [0, 0]))

                # Final distance to goal
                goals = [(19, 19), (0, 19), (19, 0)]
                min_dist = min(np.linalg.norm(agent_pos - np.array(goal)) for goal in goals)

                self.episode_count += 1
                self.current_rewards[i] = 0.0

                # Log to TensorBoard
                self.logger.record("custom/episode_reward", ep_reward)
                self.logger.record("custom/collisions", collisions)
                self.logger.record("custom/steps", steps)
                self.logger.record("custom/final_distance", min_dist)

        return True
