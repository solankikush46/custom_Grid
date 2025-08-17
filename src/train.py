# train.py

import os
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.MineEnv import MineEnv
from src.constants import SAVE_DIR
from src.utils import *
from src.wrappers import TimeStackObservation
from src.attention import AttentionCNNExtractor
from src.cnn_feature_extractor import GridCNNExtractor
from src.reward_functions import reward_d  # add more and register below

# ==============================================================
# Reward registry / resolver
# ==============================================================
_REWARD_REGISTRY = {
    "reward_d": reward_d,
    # "reward_a": reward_a,
    # "reward_b": reward_b,
}

def resolve_reward_fn(spec):
    """
    `spec` may be a string key (e.g., 'reward_d') or a callable.
    Returns a callable (reward_fn(env, new_pos) -> (reward, subrewards)).
    """
    if callable(spec):
        return spec
    if isinstance(spec, str) and spec in _REWARD_REGISTRY:
        return _REWARD_REGISTRY[spec]
    raise ValueError(f"Unknown reward spec: {spec!r}. Known: {sorted(_REWARD_REGISTRY)}")


# ==============================================================
# Callbacks
# ==============================================================
class _CaptureLogDir(BaseCallback):
    """Capture SB3's run dir (e.g., .../PPO_3/) once training starts."""
    def __init__(self):
        super().__init__()
        self.run_dir = None

    def _on_training_start(self) -> None:
        self.run_dir = self.logger.dir

    def _on_step(self) -> bool:
        return True


class InfoToTB(BaseCallback):
    """Write selected `info` metrics to TensorBoard each step."""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue
            v = info.get("current_reward", None)
            if v is not None:
                self.logger.record("timestep/current_reward", float(v))
            v = info.get("current_battery", None)
            if v is not None:
                self.logger.record("timestep/current_battery", float(v))
            v = info.get("distance_to_goal", None)
            if v is not None:
                self.logger.record("timestep/distance_to_goal", float(v))
        return True

# ==============================================================
# Environment setup
# ==============================================================
def _make_env_thunk(experiment_folder, mode, use_planner_overlay,
                    show_miners, show_predicted, render,
                    is_cnn=False, is_att=False,
                    temporal_len=12,
                    reward_fn=None):
    def _thunk():
        env = MineEnv(
            experiment_folder=experiment_folder,
            render=render,
            show_miners=show_miners,
            show_predicted=show_predicted,
            mode=mode,
            use_planner_overlay=use_planner_overlay,
            is_cnn=is_cnn or is_att,
            is_att=is_att,
            reward_fn=(reward_fn or reward_d),
        )
        if is_att:
            env = TimeStackObservation(env, num_frames=temporal_len)
        return Monitor(env)
    return _thunk


def save(model: PPO, run_dir: str, suffix: str = "") -> str:
    """Save model.zip inside run_dir, with optional suffix like _cnn or _att."""
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"model{suffix}.zip")
    model.save(model_path)
    print(f"[OK] Saved PPO model to: {model_path}")
    return model_path

# ==============================================================
# Train / Evaluate
# ==============================================================
def train(
    metadata: dict,
    total_timesteps: int = 20_000,
    mode: str = "static",     # or "constant_rate"
    use_planner_overlay: bool = False,
    show_miners: bool = False,
    show_predicted: bool = True,
    render: bool = False,
    temporal_len: int = 4,
):
    """
    Train PPO on MineEnv, save the model into the SB3-created run dir, and return:
      (model_path, run_dir)
    """
    experiment_folder = make_experiment_name(metadata)
    arch = metadata["arch"]
    reward_key = metadata.get("reward", "reward_d")
    rfn = resolve_reward_fn(reward_key)

    tb_root = os.path.join(SAVE_DIR, experiment_folder)
    os.makedirs(tb_root, exist_ok=True)

    is_cnn = arch == "cnn"
    is_att = arch == "attn"

    env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len,
            reward_fn=rfn,
        )
    ])

    policy_kwargs = None
    if is_att:
        policy_kwargs = dict(
            features_extractor_class=AttentionCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                grid_file=None,
                temporal_len=temporal_len,
            ),
        )
    elif is_cnn:
        policy_kwargs = dict(
            features_extractor_class=GridCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                grid_file=None,
            ),
        )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tb_root,
        policy_kwargs=policy_kwargs,
    )

    cap = _CaptureLogDir()
    tb_cb = InfoToTB()
    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=CallbackList([cap, tb_cb]))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()

    run_dir = cap.run_dir or getattr(model.logger, "dir", tb_root)
    suffix = "" if arch == "mlp" else f"_{arch}"
    model_path = save(model, run_dir, suffix=suffix)
    return model_path, run_dir


def evaluate(
    exp_or_meta,                       # dict metadata OR str experiment folder
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    total_timesteps: int = 300,
    render: bool = False,
    temporal_len: int = 12,
):
    """
    Evaluate a trained PPO model. Accepts either:
      - metadata dict: {"grid": "...", "miners": int, "arch": "mlp|cnn|attn", "reward": "reward_d|..."}
      - experiment folder string: e.g. "mine_50x50__20miners__mlp__reward_d"
        (old format "mine_50x50_20miners_mlp" still works; reward defaults to reward_d)
    """
    # Normalize input
    if isinstance(exp_or_meta, dict):
        metadata = exp_or_meta
        experiment_folder = make_experiment_name(metadata)
    elif isinstance(exp_or_meta, str):
        experiment_folder = exp_or_meta
        metadata = get_metadata(experiment_folder)
    else:
        raise TypeError("exp_or_meta must be a metadata dict or an experiment name (str).")

    arch = metadata["arch"]
    reward_key = metadata.get("reward", "reward_d")
    rfn = resolve_reward_fn(reward_key)

    is_cnn = arch == "cnn"
    is_att = arch == "attn"

    # Resolve model path from the experiment folder
    model_path = latest_ppo_model_path(experiment_folder)
    run_dir, run_name = latest_ppo_run(experiment_folder, require_model=True)
    eval_env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len,
            reward_fn=rfn,
        )
    ])
    try:
        print(f"[INFO] Loading {model_path} (run {run_name})")
        model = PPO.load(model_path, env=eval_env, device="auto", print_system_info=False)

        obs = eval_env.reset()
        steps = 0
        while steps < total_timesteps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            steps += 1
            if dones[0]:
                obs = eval_env.reset()
    finally:
        eval_env.close()

# ==============================================================
# Testing Functions
# ==============================================================
def train_junk(
    metadata: dict,
    timesteps: int = 1_000,
    eval_steps: int = 1_000,
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    render=True
):
    model_path, run_dir = train(
        metadata=metadata,
        total_timesteps=timesteps,
        mode=mode,
        use_planner_overlay=use_planner_overlay,
        show_miners=show_miners,
        show_predicted=show_predicted,
        render=False
    )
    print(f"[quick_junk_run] trained + saved to: {model_path}")

    evaluate(
        metadata,
        mode=mode,
        use_planner_overlay=use_planner_overlay,
        show_miners=show_miners,
        show_predicted=show_predicted,
        total_timesteps=eval_steps,
        render=render
    )
    print(f"[quick_junk_run] eval complete for run dir: {run_dir}")
    return model_path, run_dir


def train_all(total_timesteps: int = 1_000_000):
    """
    Define a list of experiments and train each, printing relevant info.
    """
    experiments = [
        {"grid": "mine_50x50", "miners": 20, "arch": "mlp", "reward": "reward_d"},
     ]

    for meta in experiments:
        exp_name = make_experiment_name(meta)
        print(f"\n[train_all] === Training {exp_name} ===")
        print(f"  Metadata: {meta}")

        tlen = 12  # should tlen be part of metadata?
        model_path, run_dir = train(
            metadata=meta,
            total_timesteps=total_timesteps,
            mode="static",
            use_planner_overlay=False,
            show_miners=False,
            show_predicted=True,
            render=False,
            temporal_len=tlen,
        )

        print(f"  -> Model saved to: {model_path}")
        print(f"  -> Run directory: {run_dir}")

# ==============================================================
# Manual control (keyboard) — same shape as evaluate, but user drives
# ==============================================================
def manual_control(
    exp_or_meta,                       # dict metadata OR str "mine_50x50__20miners__mlp__reward_d"
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = True,
    show_predicted: bool = True,
    total_timesteps: int = 300,
    render: bool = True,
    temporal_len: int = 12,
    fps: int = 15,
):
    """
    Same shape as `evaluate`, but the action comes from WASD/Arrow keys.
    Prints reward + subrewards on each step you make.
    Quit with ESC or window close.
    """
    import time
    import numpy as np
    import pygame

    # --- normalize input ---
    if isinstance(exp_or_meta, dict):
        metadata = exp_or_meta
        experiment_folder = make_experiment_name(metadata)
    elif isinstance(exp_or_meta, str):
        experiment_folder = exp_or_meta
        metadata = get_metadata(experiment_folder)
    else:
        raise TypeError("exp_or_meta must be a metadata dict or an experiment name (str).")

    arch = metadata["arch"]
    reward_key = metadata.get("reward", "reward_d")
    rfn = resolve_reward_fn(reward_key)
    is_cnn = arch == "cnn"
    is_att = arch == "attn"

    env_vec = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len,
            reward_fn=rfn,
        )
    ])

    def _fmt(v):
        try:
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    def _log_step(step_idx, rewards, infos):
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
        r = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
        subs = info.get("subrewards")
        bits = [f"step={step_idx}", f"reward={_fmt(r)}"]
        for k in ("current_battery", "distance_to_goal"):
            if k in info:
                bits.append(f"{k}={_fmt(info[k])}")
        if isinstance(subs, dict):
            sub_str = ", ".join(f"{k}={_fmt(v)}" for k, v in subs.items())
            bits.append(f"subs[{sub_str}]")
        print(" | ".join(bits))

    try:
        obs = env_vec.reset()  # window created & first frame drawn if render=True

        # Build (dr, dc) -> action_index map from the underlying env’s action list
        try:
            actions_list = env_vec.get_attr("_ACTIONS")[0]  # e.g. [(-1,0),(1,0),...]
        except Exception:
            actions_list = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        drc_to_idx = {drc: i for i, drc in enumerate(actions_list)}

        def _poll_action_index():
            """
            Return: -1 to quit, int action index to move, or None if no movement key pressed.
            """
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return -1
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                return -1
            up    = pressed[pygame.K_w] or pressed[pygame.K_UP]
            down  = pressed[pygame.K_s] or pressed[pygame.K_DOWN]
            left  = pressed[pygame.K_a] or pressed[pygame.K_LEFT]
            right = pressed[pygame.K_d] or pressed[pygame.K_RIGHT]
            dr = (-1 if up and not down else (1 if down and not up else 0))
            dc = (-1 if left and not right else (1 if right and not left else 0))
            if dr == 0 and dc == 0:
                return None
            return drc_to_idx.get((dr, dc))

        steps = 0
        clock = pygame.time.Clock()

        while steps < total_timesteps:
            act_idx = _poll_action_index()
            if act_idx == -1:
                break  # quit
            if act_idx is None:
                # idle: keep UI responsive; env renders on step/reset
                time.sleep(0.01)
                clock.tick(max(1, int(fps)))
                continue

            action = np.array([act_idx], dtype=np.int64)  # VecEnv Discrete format
            obs, rewards, dones, infos = env_vec.step(action)  # draws inside step()
            steps += 1

            _log_step(steps, rewards, infos)

            if dones[0]:
                obs = env_vec.reset()

            clock.tick(max(1, int(fps)))

    finally:
        env_vec.close()
