from grid_env import GridWorldEnv
from episode_callback import EpisodeStatsCallback
import time
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

def run_sample_agent(episodes, env):
    for ep in range(episodes):
        print(f"\n Episode-{ep+1}: ")
        obs = env.reset()
        terminated = False

        while not terminated: 
            action =  env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render_pygame()
            time.sleep(0.1)

        env.episode_summary()

    env.close()

def PPO_train_model(time_steps):
    # Create environment
    env = GridWorldEnv()
    vec_env = make_vec_env(lambda: GridWorldEnv(), n_envs=1)

    # Logging paths
    log_path = os.path.join('logs', 'PPO_custom_grid')
    model_save_path = os.path.join("SavedModels", "PPO_custom_grid")

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
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
    env = GridWorldEnv()
    vec_env = make_vec_env(lambda: GridWorldEnv(), n_envs=1)

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
    env = GridWorldEnv()
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
                break  # stop collecting timesteps beyond the target

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

        # Final distance to nearest goal
        goals = [(19, 19), (0, 19), (19, 0)]
        agent_pos = np.array(info.get("agent_pos", [0, 0]))
        final_distance = min(np.linalg.norm(agent_pos - np.array(goal)) for goal in goals)
        collisions = info.get("collisions", -1)

        # TensorBoard logging — use timestep_count as the x-axis
        writer.add_scalar("custom/episode_reward", total_reward, timestep_count)
        writer.add_scalar("custom/steps", steps, timestep_count)
        writer.add_scalar("custom/final_distance", final_distance, timestep_count)
        writer.add_scalar("custom/collisions", collisions, timestep_count)

        # Local logging (optional)
        rewards_log.append(total_reward)
        steps_log.append(steps)
        collisions_log.append(collisions)
        final_distance_log.append(final_distance)

        if episode_count % 100 == 0:
            print(f"[Q] Ep {episode_count} | Total Reward: {total_reward:.2f} | Steps: {steps} | ε={agent.epsilon:.3f} | T={timestep_count}")

    # Save Q-table and stats
    agent.save_q_table(os.path.join(log_path, "q_table.npy"))
    """
    np.save(os.path.join(log_path, "q_rewards.npy"), rewards_log)
    np.save(os.path.join(log_path, "q_steps.npy"), steps_log)
    np.save(os.path.join(log_path, "q_collisions.npy"), collisions_log)
    np.save(os.path.join(log_path, "q_distances.npy"), final_distance_log)
    """




def evaluate_Model(model, n_eval_episodes=5, sleep_time=0.1):
    total_rewards = []
    final_env = None  # To store the last environment's state

    for ep in range(n_eval_episodes):
        env = GridWorldEnv()
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
            final_env = env  # Save the last episode for stats

        env.close()

    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n Evaluation complete over {n_eval_episodes} episodes")
    print(f" Mean Reward: {mean_reward:.2f}")
    print("\n Final Episode Summary:")
    final_env.episode_summary()

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



if __name__ == "__main__":


    test_env =  GridWorldEnv()
    run_sample_agent(3, test_env)
    """
    #Training and evaluating PPO model

    #PPO_train_model(500000)
    PPO_model = PPO.load("SavedModels/PPO_custom_grid.zip", env=test_env)
    evaluate_Model(PPO_model)
    """
    '''
    #Training and evaluating DQN model

    #DQN_train_model(500000)
    DQN_model = DQN.load("SavedModels/DQN_custom_grid.zip", env=test_env)
    evaluate_Model(DQN_model)
    '''

    """
    agent = QLearningAgent(grid_size=20, action_size=8)
    log_dir = "logs/Q_custom_grid"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    train_Q_agent(agent, writer, log_path=log_dir, total_timesteps = 500000, max_steps=200)
    writer.close()
    agent.load_q_table(os.path.join(log_dir, "q_table.npy"))
    evaluate_Q_agent(test_env, agent)
    """

