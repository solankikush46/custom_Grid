# train.py

from grid_env import GridWorldEnv
from episode_callback import EpisodeStatsCallback
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from Qlearning import QLearningAgent
from torch.utils.tensorboard import SummaryWriter
import time
from constants import *

N_SENSORS = 4 

def run_sample_agent(episodes, env):
    for ep in range(episodes):
        print(f"\n Episode-{ep+1}: ")
        obs, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated): 
            action =  env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render_pygame()
            time.sleep(0.1)

        env.episode_summary()

    env.close()

def PPO_train_model(time_steps):
    # Create environment
    def make_env():
        return GridWorldEnv(grid_height=20, grid_width=20, n_obstacles=40, n_sensors=N_SENSORS)

    env = make_env()
    vec_env = make_vec_env(make_env, n_envs=1)

    # Logging paths
    log_path = os.path.join('logs', 'PPO_custom_grid')
    model_save_path = os.path.join("SavedModels", "PPO_custom_grid")

    # Create the PPO model
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_path
    )

    callback = EpisodeStatsCallback()

    # Train the model with callback
    model.learn(total_timesteps=time_steps, callback=callback)

    # Save the trained model
    model.save(model_save_path)

    print("\n PPO training complete and metrics logged to TensorBoard.")
    return model

def DQN_train_model(time_steps):
    def make_env():
        return GridWorldEnv(grid_height=20, grid_width=20, n_obstacles=40, n_sensors=N_SENSORS)

    env = make_env()
    vec_env = make_vec_env(make_env, n_envs=1)

    log_path = os.path.join('logs', 'DQN_custom_grid')
    model_save_path = os.path.join("SavedModels", "DQN_custom_grid")

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_path
    )

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=time_steps, callback=callback)
    model.save(model_save_path)

    print("\n Training complete and metrics logged to TensorBoard.")
    return model

def train_Q_agent(agent, writer, log_path, total_timesteps=500_000, max_steps=200):
    env = GridWorldEnv(grid_height=20, grid_width=20, n_obstacles=40, n_sensors=N_SENSORS)
    timestep_count = 0
    episode_count = 0

    rewards_log = []
    steps_log = []
    collisions_log = []
    final_distance_log = []

    while timestep_count < total_timesteps:
        state, _ = env.reset()
        state_idx = agent.state_to_index(state)
        total_reward = 0
        steps = 0

        episode_timesteps = 0
        done = False

        while steps < max_steps and not done:
            if timestep_count >= total_timesteps:
                break

            action = agent.choose_action(state_idx)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state_idx = agent.state_to_index(next_state)

            agent.update(state_idx, action, reward, next_state_idx)
            state_idx = next_state_idx

            total_reward += reward
            steps += 1
            timestep_count += 1
            done = terminated or truncated

        agent.decay_epsilon(episode_count)
        episode_count += 1

        goals = [(19, 19), (0, 19), (19, 0)]
        agent_pos = np.array(info.get("agent_pos", [0, 0]))
        final_distance = min(np.linalg.norm(agent_pos - np.array(goal)) for goal in goals)
        collisions = info.get("collisions", -1)

        writer.add_scalar("custom/episode_reward", total_reward, timestep_count)
        writer.add_scalar("custom/steps", steps, timestep_count)
        writer.add_scalar("custom/final_distance", final_distance, timestep_count)
        writer.add_scalar("custom/collisions", collisions, timestep_count)

        rewards_log.append(total_reward)
        steps_log.append(steps)
        collisions_log.append(collisions)
        final_distance_log.append(final_distance)

        if episode_count % 100 == 0:
            print(f"[Q] Ep {episode_count} | Total Reward: {total_reward:.2f} | Steps: {steps} | ε={agent.epsilon:.3f} | T={timestep_count}")

    agent.save_q_table(os.path.join(log_path, "q_table.npy"))

'''
def evaluate_model(model, n_eval_episodes=5, sleep_time=0.1):
    total_rewards = []
    final_env = None

    for ep in range(n_eval_episodes):
        env = GridWorldEnv(grid_height=20, grid_width=20, n_obstacles=40, n_sensors=N_SENSORS)
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        print(f"\n--- Episode {ep + 1} ---")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render_pygame()
            time.sleep(sleep_time)
            episode_reward += reward

        total_rewards.append(episode_reward)

        if ep == n_eval_episodes - 1:
            final_env = env

        env.close()

    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n Evaluation complete over {n_eval_episodes} episodes")
    print(f" Mean Reward: {mean_reward:.2f}")
    print("\n Final Episode Summary:")
    final_env.episode_summary()
'''

def evaluate_Q_agent(env, agent, n_episodes=3, delay=0.1):
    print("\nEvaluating trained agent...\n")
    for eval_ep in range(n_episodes):
        state, _ = env.reset()
        state_idx = agent.state_to_index(state)
        done = False
        time.sleep(1)
        while not done:
            env.render()
            time.sleep(delay)
            action = np.argmax(agent.q_table[state_idx])
            next_state, reward, terminated, truncated, _ = env.step(action)
            state_idx = agent.state_to_index(next_state)
            done = terminated or truncated
        env.episode_summary()

##==============================================================
## Cole's Experiments
##==============================================================
def train_PPO_model(env: gym.Env, timesteps: int):
    # create environment
    # why creating two GridWorldEnv???
    vec_env = make_vec_env(lambda: GridWorldEnv(20, 20, 0, 0), n_envs=1)

    # logging paths
    log_path = os.path.join('logs', 'PPO_custom_grid')
    model_save_path = os.path.join("SavedModels", "PPO_custom_grid")

    # create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_path
        )

    callback = EpisodeStatsCallback()

    # train the model with callback
    model.learn(total_timesteps=timesteps, callback=callback)

    # save trained model
    model.save(model_save_path)
    print("\n PPO training complete and metrics logged to TensorBoard.")

    return model

def evaluate_model(env: gym.Env, model, n_eval_episodes=5, sleep_time=0.1):
    total_rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # stable baselines3 sometimes returns tuple
            obs = obs[0]

        terminated, truncated = False, False
        episode_reward = 0
        print(f"\n--- Episode {ep + 1} ---")

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            env.render_pygame()
            time.sleep(sleep_time)
            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n Evaluation complete over {n_eval_episodes} episodes")
    print(f" Mean Reward: {mean_reward:.2f}")
    print("\n Final Episode Summary:")
    env.episode_summary()

def load_model_and_evaluate(model_key: str, env, n_eval_episodes=5, sleep_time=0.1, render: bool = True):
    """
    Load a PPO model and evaluate it on the provided environment.

    Args:
        model_key (str): Key name of the model in config.MODELS
        env (gym.Env): The environment instance for evaluation.
        n_eval_episodes (int): Number of episodes to run.
        sleep_time (float): Delay between steps for rendering.
        render (bool): Whether to render the environment using pygame.
    """
    model_path = MODELS.get(model_key)
    if not model_path:
        raise ValueError(f"Unknown model key: {model_key}")

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file not found at: {model_path}.zip")

    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env)

    total_rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        terminated = False
        truncated = False
        episode_reward = 0

        print(f"\n--- Episode {ep + 1} ---")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render_pygame()
                time.sleep(sleep_time)
            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n Evaluation complete over {n_eval_episodes} episodes")
    print(f" Mean Reward: {mean_reward:.2f}")
    print("\n Final Episode Summary:")
    env.episode_summary()
