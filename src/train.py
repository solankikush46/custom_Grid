# train.py

import os
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.MineEnv import MineEnv
from src.constants import SAVE_DIR
from src.utils import latest_ppo_model_path, latest_ppo_run

from src.wrappers import TimeStackObservation
from src.attention import AttentionCNNExtractor
from src.cnn_feature_extractor import GridCNNExtractor

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
# Metadata helpers
# ==============================================================
def make_experiment_name(metadata: dict) -> str:
    """
    metadata = {
        "grid": "50x50",
        "miners": 12,
        "arch": "cnn" | "attn" | "mlp"
    }
    """
    return f"{metadata['grid']}_{metadata['miners']}miners_{metadata['arch']}"


def get_metadata(experiment_name: str) -> dict:
    """
    Inverse of make_experiment_name.
    e.g. '50x50_12miners_cnn' → {"grid": "50x50", "miners": 12, "arch": "cnn"}
    """
    parts = experiment_name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Bad experiment name: {experiment_name}")

    grid = parts[0]
    miners_str = parts[1]
    if not miners_str.endswith("miners"):
        raise ValueError(f"Bad miners spec in {experiment_name}")
    miners = int(miners_str.replace("miners", ""))
    arch = parts[2]

    return {"grid": grid, "miners": miners, "arch": arch}

# ==============================================================
# Environment setup
# ==============================================================
def _make_env_thunk(experiment_folder, mode, use_planner_overlay,
                    show_miners, show_predicted, render: bool,
                    is_cnn: bool = False, is_att: bool = False,
                    temporal_len: int = 12):
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
    render=False,
    temporal_len: int = 4,
):
    """
    Train PPO on MineEnv, save the model into the SB3-created run dir, and return:
      (model_path, run_dir)
    """
    experiment_folder = make_experiment_name(metadata)
    arch = metadata["arch"]

    tb_root = os.path.join(SAVE_DIR, experiment_folder)
    os.makedirs(tb_root, exist_ok=True)

    is_cnn = arch == "cnn"
    is_att = arch == "attn"

    env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len
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
    metadata: dict,
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    total_timesteps: int = 300,
    render=True,
    temporal_len: int = 12,
):
    experiment_folder = make_experiment_name(metadata)
    arch = metadata["arch"]

    model_path = latest_ppo_model_path(experiment_folder)
    run_dir, run_name = latest_ppo_run(experiment_folder, require_model=True)

    is_cnn = arch == "cnn"
    is_att = arch == "attn"

    eval_env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len
        )
    ])
    try:
        print(f"[INFO] Loading {model_path} (run {run_name})")
        model = PPO.load(model_path, env=eval_env,
                         device="auto", print_system_info=False)
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
        metadata=metadata,
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
        {"grid": "mine_50x50", "miners": 20, "arch": "mlp"},
     ]

    for meta in experiments:
        exp_name = make_experiment_name(meta)
        print(f"\n[train_all] === Training {exp_name} ===")
        print(f"  Metadata: {meta}")

        tlen = 12 # should tlen be part of metadata?
        model_path, run_dir = train(
            metadata=meta,
            total_timesteps=total_timesteps,
            # tweak these if you want different global defaults:
            mode="static",
            use_planner_overlay=False,
            show_miners=False,
            show_predicted=True,
            render=True,
            temporal_len=tlen,
        )

        print(f"  -> Model saved to: {model_path}")
        print(f"  -> Run directory: {run_dir}")

def test_manual_control():
    """
    Launch a manual-control test session.
    Allows user to move the miner with keyboard input and prints
    per-step rewards + subrewards to console.
    """
    # Pick any experiment folder you want to test
    experiment_folder = "mine_50x50_12miners_mlp"

    env = MineEnv(
        experiment_folder=experiment_folder,
        render=True,
        use_planner_overlay=True,
        manual_control=True
    )
