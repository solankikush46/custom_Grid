# wrappers.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class TimeStackObservation(gym.Wrapper):
    """
    Stack the last T frames along a leading time axis.
    Base env must emit Box(C, H, W) floats. Output is Box(T, C, H, W).
    """
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        self.num_frames = num_frames

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("TimeStackObservation requires a Box observation_space")
        if len(obs_space.shape) != 3:
            raise ValueError(f"Expected 3D (C,H,W) obs, got {obs_space.shape}")

        C, H, W = obs_space.shape

        # Compute scalar bounds, then expand to full shape for safety
        low_val = float(np.min(obs_space.low)) if np.ndim(obs_space.low) else float(obs_space.low)
        high_val = float(np.max(obs_space.high)) if np.ndim(obs_space.high) else float(obs_space.high)

        self.observation_space = spaces.Box(
            low=np.full((num_frames, C, H, W), low_val, dtype=np.float32),
            high=np.full((num_frames, C, H, W), high_val, dtype=np.float32),
            dtype=np.float32,
        )

        self._frames = deque(maxlen=num_frames)

    # Gymnasium API: return (obs, info)
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames.clear()
        self._push(obs)
        # Bootstrap stack with the first frame
        while len(self._frames) < self.num_frames:
            self._frames.append(self._frames[-1])
        return self._stack(), info

    # Gymnasium API: return (obs, reward, terminated, truncated, info)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._push(obs)
        return self._stack(), reward, terminated, truncated, info

    # ---- internals ----
    def _push(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected obs with shape (C,H,W), got {arr.shape}")
        self._frames.append(arr)

    def _stack(self):
        # (T, C, H, W)
        return np.stack(self._frames, axis=0)
