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

class _CaptureLogDir(BaseCallback):
    """Capture SB3's run dir (e.g., .../PPO_3/) once training starts."""
    def __init__(self):
        super().__init__()
        self.run_dir = None

    def _on_training_start(self) -> None:
        self.run_dir = self.logger.dir  # set by SB3

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
            # Optional: distance as well
            v = info.get("distance_to_goal", None)
            if v is not None:
                self.logger.record("timestep/distance_to_goal", float(v))
        return True

##==============================================================
## Training Helpers
##==============================================================
def _make_env_thunk(experiment_folder, mode, use_planner_overlay, show_miners, show_predicted, render: bool, is_cnn: bool = False, is_att: bool = False, temporal_len: int = 12):
    def _thunk():
        env = MineEnv(
            experiment_folder=experiment_folder,
            render=render,
            show_miners=show_miners,
            show_predicted=show_predicted,
            mode=mode,
            use_planner_overlay=use_planner_overlay,
            is_cnn=is_cnn or is_att,  # attention uses CNN frames too
            is_att=is_att,
        )
        if is_att:
            env = TimeStackObservation(env, num_frames=temporal_len)  # shape (T,C,H,W) :contentReference[oaicite:11]{index=11}
        return Monitor(env)  # SB3 will still log TB in run_dir
    return _thunk


def save(model: PPO, run_dir: str, suffix: str = "") -> str:
    """Save model.zip inside run_dir, with optional suffix like _cnn or _att."""
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"model{suffix}.zip")
    model.save(model_path)
    print(f"[OK] Saved PPO model to: {model_path}")
    return model_path

def train(
    experiment_folder: str,
    total_timesteps: int = 20_000,
    mode: str = "static",               # or "constant_rate"
    use_planner_overlay: bool = False,
    show_miners: bool = False,
    show_predicted: bool = True,
    render=False,
    is_cnn: bool = False,
    is_att: bool = False,
    temporal_len: int = 4,
):
    """
    Train PPO on MineEnv, save the model into the SB3-created run dir, and return:
      (model_path, run_dir)
    TensorBoard root is SAVE_DIR/<exp>/; SB3 creates .../PPO_<n>/ inside it.
    """
    tb_root = os.path.join(SAVE_DIR, experiment_folder)
    os.makedirs(tb_root, exist_ok=True)

    env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len
        )
    ])

    policy_kwargs = None

    if is_att:
        # Temporal Attention over stacked frames
        policy_kwargs = dict(
            features_extractor_class=AttentionCNNExtractor,   # :contentReference[oaicite:5]{index=5}
            features_extractor_kwargs=dict(
                features_dim=128,
                grid_file=None,       # optional hint to your backbone
                temporal_len=temporal_len,
            ),
        )
    elif is_cnn:
        # Plain CNN → vector for MLP policy head
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
        model.learn(total_timesteps=total_timesteps, callback=CallbackList([cap, tb_cb]))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()

    # The exact SB3 run directory (e.g., .../PPO_3)
    run_dir = cap.run_dir or getattr(model.logger, "dir", tb_root)

    suffix = ""
    if is_att:
        suffix = "_att"
    elif is_cnn:
        suffix = "_cnn"

    model_path = save(model, run_dir, suffix = suffix)
    return model_path, run_dir

def evaluate(
    experiment_folder: str,
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    total_timesteps: int = 300,
    render=True,
    is_cnn: bool = False,
    is_att: bool = False,
    temporal_len: int = 12,
):
    """
    Find latest PPO_<n>/model.zip under SAVE_DIR/<experiment_folder>, load it,
    and run a short rollout (rendered if render=True).
    """
    model_path = latest_ppo_model_path(experiment_folder)
    run_dir, run_name = latest_ppo_run(experiment_folder, require_model=True)

    eval_env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            is_cnn=is_cnn, is_att=is_att, temporal_len=temporal_len
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

##==============================================================
## Training Test Functions
##==============================================================
def train_junk(
    experiment_folder: str,
    timesteps: int = 1_000,
    eval_steps: int = 1_000,
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    render=True
):
    """
    Fast smoke test: train a tiny PPO for a few steps and immediately evaluate.
    Saves the model in the SB3-created PPO_<n> folder for this experiment.

    Returns:
        (model_path, run_dir)
    """
    # 1) quick train + save (train() already saves the model)
    model_path, run_dir = train(
        experiment_folder=experiment_folder,
        total_timesteps=timesteps,
        mode=mode,
        use_planner_overlay=use_planner_overlay,
        show_miners=show_miners,
        show_predicted=show_predicted,
        render=False  # typically keep training headless
    )
    print(f"[quick_junk_run] trained + saved to: {model_path}")

    # 2) quick eval (loads latest PPO_<n>/model.zip automatically)
    evaluate(
        experiment_folder=experiment_folder,
        mode=mode,
        use_planner_overlay=use_planner_overlay,
        show_miners=show_miners,
        show_predicted=show_predicted,
        total_timesteps=eval_steps,
        render=render
    )
    print(f"[quick_junk_run] eval complete for run dir: {run_dir}")
    return model_path, run_dir
