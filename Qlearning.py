import numpy as np
import random

class QLearningAgent:
    def __init__(self, grid_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_size = action_size
        self.q_table = np.zeros((self.state_size, self.action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def state_to_index(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_index])

    def update(self, state_idx, action, reward, next_state_idx):
        best_next_action = np.max(self.q_table[next_state_idx])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-self.epsilon_decay * episode)

    def save_q_table(self, path):
        np.save(path, self.q_table)

    def load_q_table(self, path):
        self.q_table = np.load(path)
