# train.py

import os
import traceback
import time

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
def _make_env_thunk(
    experiment_folder,
    mode,
    use_planner_overlay,
    show_miners,
    show_predicted,
    render,
    arch="mlp",
    temporal_len=12,
    reward_fn=None,
):
    """
    Returns a callable that creates a single MineEnv instance wrapped with Monitor.

    Rules (strict, no fallbacks):
      - If arch starts with 'a' (attention), TimeStackObservation MUST exist and will be
        applied to produce (T, C, H, W). temporal_len MUST be > 1.
      - We set _has_time_stack and _tstack_len on the Monitor wrapper (no try/except).
      - Any failure raises immediately.
    """
    def _thunk():
        from stable_baselines3.common.monitor import Monitor
        from .MineEnv import MineEnv
        from .reward_functions import reward_d

        s_arch = str(arch or "").lower()
        is_att = s_arch.startswith("a")

        # Build base env
        env = MineEnv(
            experiment_folder=experiment_folder,
            render=render,
            show_miners=show_miners,
            show_predicted=show_predicted,
            mode=mode,
            use_planner_overlay=use_planner_overlay,
            arch=s_arch,
            reward_fn=(reward_fn or reward_d),
        )

        used_time_stack = False
        if is_att:
            if temporal_len is None or int(temporal_len) <= 1:
                raise ValueError(
                    f"Attention arch '{s_arch}' requires temporal_len > 1; got {temporal_len}."
                )
            # Require the project wrapper; do not fall back.
            from .wrappers import TimeStackObservation  # will raise ImportError if missing
            env = TimeStackObservation(env, num_frames=int(temporal_len))  # must yield (T,C,H,W)
            used_time_stack = True

        wrapped = Monitor(env)
        # Stamp metadata for train() assertions (no try/except)
        wrapped._has_time_stack = bool(used_time_stack)
        wrapped._tstack_len = int(temporal_len)
        wrapped._arch = s_arch
        return wrapped

    return _thunk

def save(model: PPO, run_dir: str, suffix: str = "") -> str:
    """Save model.zip inside run_dir, with optional suffix like _c0/_a1/etc."""
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, f"model{suffix}.zip")
    model.save(model_path)
    print(f"[OK] Saved PPO model to: {model_path}")
    return model_path

# ==============================================================
# Train / Evaluate
# ==============================================================
# ==============================================================
# Train (no fallbacks; fail fast)
# ==============================================================
def train(
    metadata: dict,
    total_timesteps: int = 20_000,
    mode: str = "static",     # or "constant_rate"
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    render: bool = False,
    temporal_len: int = 4,
):
    """
    Train PPO on MineEnv, save the model into the SB3-created run dir, and return:
      (model_path, run_dir)

    Arch scheme (strict):
      - 'mlp' -> vector obs via MineEnv._make_obs
      - 'c0','c1' -> CNN extractor over (C,H,W)
      - 'a0','a1' -> Attention extractor over (T,C,H,W) via TimeStackObservation only.
                     VecFrameStack is NOT used; absence of timestack raises.
    """
    # ---- classify arch once ----
    arch = str(metadata.get("arch", "mlp")).lower()
    is_att = arch.startswith("a")
    is_cnn = is_att or arch.startswith("c")
    variant = arch[1:] if is_cnn and len(arch) > 1 else None

    # ---- experiment + logging roots ----
    experiment_folder = make_experiment_name(metadata)
    reward_key = metadata.get("reward", "reward_d")
    rfn = resolve_reward_fn(reward_key)
    tb_root = os.path.join(SAVE_DIR, experiment_folder)
    os.makedirs(tb_root, exist_ok=True)

    # ---- build vec env ----
    env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            arch=arch, temporal_len=temporal_len, reward_fn=rfn,
        )
    ])

    # ---- assertions for attention timestack ----
    if is_att:
        if not hasattr(env.envs[0], "_has_time_stack"):
            raise RuntimeError("Attention arch requires TimeStackObservation; _has_time_stack not set on env.")
        if not env.envs[0]._has_time_stack:
            raise RuntimeError("Attention arch detected, but TimeStackObservation was not applied.")
        if not hasattr(env.envs[0], "_tstack_len"):
            raise RuntimeError("TimeStackObservation did not stamp _tstack_len on env wrapper.")
        if int(env.envs[0]._tstack_len) != int(temporal_len):
            raise RuntimeError(
                f"Time stack length mismatch: env={env.envs[0]._tstack_len}, requested={temporal_len}."
            )

    # ---- choose extractor + kwargs
    if is_att:
        policy_kwargs = dict(
            features_extractor_class=AttentionCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                temporal_len=int(temporal_len),
                #variant=variant,
            ),
        )
    elif is_cnn:
        policy_kwargs = dict(
            features_extractor_class=GridCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                #variant=variant,
            ),
        )
    else:
        policy_kwargs = None  # pure MLP

    # build model
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
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([cap, tb_cb]),
        )
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
    load_model_path: str = None,       # NEW: absolute path to model.zip for a specific run
):
    """
    Evaluate a trained PPO model. Accepts either:
      - metadata dict: {"grid": "...", "miners": int, "arch": "mlp|c*|a*", "reward": "reward_d|..."}
      - experiment folder string: e.g. "mine_50x50__20miners__a1__reward_d"

    If `load_model_path` is provided, that specific checkpoint is loaded (overrides latest).
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

    arch = str(metadata["arch"]).lower()
    reward_key = metadata.get("reward", "reward_d")
    rfn = resolve_reward_fn(reward_key)

    is_att = arch.startswith("a")
    is_cnn = is_att or arch.startswith("c")

    # Resolve model + run info
    if load_model_path:
        model_path = load_model_path
        run_dir = os.path.dirname(model_path)
        run_name = os.path.basename(run_dir)
        load_base = model_path[:-4] if model_path.endswith(".zip") else model_path
    else:
        model_path = latest_ppo_model_path(experiment_folder)
        run_dir, run_name = latest_ppo_run(experiment_folder, require_model=True)
        load_base = model_path[:-4] if model_path.endswith(".zip") else model_path

    # Build eval env that matches the arch (attention uses TimeStackObservation)
    eval_env = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            arch=arch, temporal_len=temporal_len,
            reward_fn=rfn,
        )
    ])

    # ---- DEBUG: show observation/action spaces to verify correctness ----
    def _desc_space(space):
        try:
            cls = space.__class__.__name__
            shp = getattr(space, "shape", None)
            dtype = getattr(space, "dtype", None)
            if shp is not None:
                return f"{cls}(shape={tuple(shp)}, dtype={dtype})"
            return f"{cls}"
        except Exception as e:
            return f"<unprintable space: {e}>"

    try:
        print(f"[DEBUG] arch={arch!r} | is_cnn={is_cnn} | is_att={is_att} | temporal_len={temporal_len}")
        print(f"[DEBUG] VecEnv observation_space: {_desc_space(eval_env.observation_space)}")
        print(f"[DEBUG] VecEnv action_space:      {_desc_space(eval_env.action_space)}")
        try:
            raw_env = eval_env.envs[0]
            print(f"[DEBUG] Raw env observation_space: {_desc_space(raw_env.observation_space)}")
            print(f"[DEBUG] Raw env action_space:      {_desc_space(raw_env.action_space)}")
            if hasattr(raw_env, "_has_time_stack"):
                print(f"[DEBUG] _has_time_stack={raw_env._has_time_stack}  "
                      f"_tstack_len={getattr(raw_env, '_tstack_len', None)}")
        except Exception:
            pass

        print(f"[INFO] Loading {load_base} (run {run_name})")
        model = PPO.load(load_base, env=eval_env, device="auto", print_system_info=False)

        # After load: print what the checkpoint expects
        try:
            print(f"[DEBUG] Loaded model observation_space: {_desc_space(getattr(model, 'observation_space', None))}")
            print(f"[DEBUG] Loaded model action_space:      {_desc_space(getattr(model, 'action_space', None))}")
        except Exception:
            pass

        if total_timesteps <= 0:
            return  # just inspect/print

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
# Testing / batch training helpers
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
    Define a list of experiments and train each. By default trains for 1,000,000 steps.
    Includes an a1 (attention over c1) configuration.
    """
    experiments = [
        {"grid": "mine_50x50", "miners": 20, "arch": "a1", "reward": "reward_d"},
    ]

    for meta in experiments:
        exp_name = make_experiment_name(meta)
        print(f"\n[train_all] === Training {exp_name} ===")
        print(f"  Metadata: {meta}")

        tlen = 12  # temporal length for attention models
        model_path, run_dir = train(
            metadata=meta,
            total_timesteps=total_timesteps,
            mode="static",
            use_planner_overlay=True,
            show_miners=False,
            show_predicted=True,
            render=False,
            temporal_len=tlen,
        )

        print(f"  -> Model saved to: {model_path}")
        print(f"  -> Run directory: {run_dir}")

def _find_latest_run_and_model(exp_dir: str):
    """
    exp_dir = absolute path to saved_experiments/<experiment>/
    Returns (run_dir, model_zip_path) or (None, None) if not found.
    Prefers PPO_* subdirs; falls back to any model*.zip under exp_dir.
    """
    # Prefer PPO_* directories, pick the latest by natural sort
    ppo_runs = []
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if os.path.isdir(full) and name.startswith("PPO_"):
            ppo_runs.append((name, full))
    if ppo_runs:
        # sort by the numeric suffix if present
        def _k(t):
            n = t[0].split("_", 1)[-1]
            return int(n) if n.isdigit() else 0
        ppo_runs.sort(key=_k)
        # scan from newest to oldest for a model zip
        for _, run_dir in reversed(ppo_runs):
            zips = [f for f in os.listdir(run_dir) if f.startswith("model") and f.endswith(".zip")]
            if zips:
                zips.sort()
                return run_dir, os.path.join(run_dir, zips[-1])

    # Fallback: search recursively for any model*.zip
    for dirpath, _, filenames in os.walk(exp_dir):
        zips = [f for f in filenames if f.startswith("model") and f.endswith(".zip")]
        if zips:
            zips.sort()
            return dirpath, os.path.join(dirpath, zips[-1])

    return None, None

def evaluate_all(
    base_dir: str = SAVE_DIR,
    episodes: int = 5,
    render: bool = True,
    verbose: bool = False,
    deterministic: bool = True,
    temporal_len: int = 12,
    include: list = None,
    exclude: list = None,
):
    include = include or []
    exclude = exclude or []

    if not os.path.isdir(base_dir):
        print(f"[evaluate_all] No such directory: {base_dir}")
        return []

    def _model_file_in(run_dir):
        p0 = os.path.join(run_dir, "model.zip")
        if os.path.isfile(p0):
            return p0
        for fn in os.listdir(run_dir):
            if fn.startswith("model_") and fn.endswith(".zip"):
                return os.path.join(run_dir, fn)
        return None

    def _runs_in(exp_path):
        try:
            names = sorted(
                os.listdir(exp_path),
                key=lambda n: int(n.split("_")[-1]) if n.startswith("PPO_") else -1,
                reverse=True,
            )
        except Exception:
            names = os.listdir(exp_path)
        for nm in names:
            rd = os.path.join(exp_path, nm)
            if os.path.isdir(rd) and nm.startswith("PPO_"):
                yield rd

    def _should_skip(path):
        if include and not any(tok in path for tok in include):
            return True
        if exclude and any(tok in path for tok in exclude):
            return True
        return False

    experiments = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    if not experiments:
        print(f"[evaluate_all] No experiments under {base_dir}")
        return []

    all_episode_metrics = []
    evaluated = []

    for exp_name in experiments:
        exp_path = os.path.join(base_dir, exp_name)
        if _should_skip(exp_path):
            continue

        try:
            meta = get_metadata(exp_name)
            arch = meta.get("arch", "mlp")
            reward_key = meta.get("reward", "reward_d")
        except Exception as e:
            print(f"[evaluate_all] Skipping {exp_name}: cannot parse metadata ({e})")
            continue

        print(f"\n[evaluate_all] === {exp_name} ===")

        for run_dir in _runs_in(exp_path):
            model_zip = _model_file_in(run_dir)
            if not model_zip:
                print(f"  [skip] {run_dir}: no model zip found")
                continue

            print(f"  run_dir: {run_dir}")
            print(f"  model:   {model_zip}")
            print(f"  arch:    {arch} | reward: {reward_key}")

            # 1) Delegate to `evaluate` to build the env exactly like training and PRINT spaces
            try:
                evaluate(
                    meta,
                    mode="static",
                    use_planner_overlay=True,
                    show_miners=False,
                    show_predicted=True,
                    total_timesteps=0,             # just print spaces + load
                    render=render,
                    temporal_len=temporal_len,
                    load_model_path=model_zip,     # <-- key: pick THIS run's model
                )
            except Exception as e:
                print(f"  [warn] evaluate() failed for {run_dir}: {e}")

            # 2) Optionally, still run episodes here for metrics (same as before)
            rfn = resolve_reward_fn(reward_key)
            try:
                env = DummyVecEnv([
                    _make_env_thunk(
                        experiment_folder=exp_name,
                        mode="static",
                        use_planner_overlay=True,
                        show_miners=False,
                        show_predicted=True,
                        render=render,
                        arch=arch,
                        temporal_len=temporal_len,
                        reward_fn=rfn,
                    )
                ])
            except Exception as e:
                print(f"  [error] could not construct env: {e}")
                continue

            load_base = model_zip[:-4] if model_zip.endswith(".zip") else model_zip
            try:
                model = PPO.load(load_base, env=env, device="auto", print_system_info=False)
            except Exception as e:
                print(f"  [error] could not load model: {e}")
                try: env.close()
                except Exception: pass
                continue

            ep_metrics = []
            try:
                episodes_done = 0
                obs = env.reset()
                ep_return = 0.0
                ep_len = 0

                while episodes_done < episodes:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, rewards, dones, infos = env.step(action)

                    r = float(rewards[0]) if isinstance(rewards, (list, tuple, np.ndarray)) else float(rewards)
                    info = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
                    ep_return += r
                    ep_len += 1

                    if verbose:
                        print(f"    t={ep_len}  reward={r:.4f}")
                        subs = info.get("subrewards") or {}
                        if isinstance(subs, dict):
                            for k, v in subs.items():
                                try:
                                    print(f"      sub[{k}]: {float(v):.6f}")
                                except Exception:
                                    print(f"      sub[{k}]: {v}")

                    if dones[0]:
                        m = {
                            "episode_return": float(ep_return),
                            "episode_length": int(ep_len),
                            "obstacle_hits": int(info.get("obstacle_hits", 0)),
                            "revisit_count": int(info.get("revisit_count", 0)),
                            "avg_battery": float(info.get("average_battery_level", 0.0)),
                            "reached_goal": bool(info.get("terminated", False)),
                        }
                        ep_metrics.append(m)
                        all_episode_metrics.append({**m, "experiment": exp_name, "run_dir": run_dir})
                        episodes_done += 1
                        ep_return = 0.0
                        ep_len = 0
                        obs = env.reset()

                if ep_metrics:
                    import numpy as np
                    er = [m["episode_return"] for m in ep_metrics]
                    el = [m["episode_length"] for m in ep_metrics]
                    sr = [1.0 if m["reached_goal"] else 0.0 for m in ep_metrics]
                    print(f"  [run summary] episodes={len(ep_metrics)} | "
                          f"mean_return={np.mean(er):.3f} | mean_len={np.mean(el):.1f} | "
                          f"success_rate={np.mean(sr):.2%}")
                else:
                    print("  [run summary] no completed episodes")

            except KeyboardInterrupt:
                print("  [interrupted]")
            except Exception as e:
                print(f"  [error during eval] {e}")
            finally:
                try: env.close()
                except Exception: pass

            evaluated.append((exp_name, run_dir, model_zip))

    if not all_episode_metrics:
        print("\n[evaluate_all] No completed episodes across all runs.")
        return evaluated

    import numpy as np
    ER = np.array([m["episode_return"] for m in all_episode_metrics], dtype=float)
    EL = np.array([m["episode_length"] for m in all_episode_metrics], dtype=float)
    SR = np.array([1.0 if m["reached_goal"] else 0.0 for m in all_episode_metrics], dtype=float)
    HITS = np.array([m["obstacle_hits"] for m in all_episode_metrics], dtype=float)
    RV = np.array([m["revisit_count"] for m in all_episode_metrics], dtype=float)
    AB = np.array([m["avg_battery"] for m in all_episode_metrics], dtype=float)

    print("\n[evaluate_all] ===== Cross-Run Summary =====")
    print(f"  total_episodes:   {len(all_episode_metrics)}")
    print(f"  mean_return:      {ER.mean():.3f} (±{ER.std():.3f})")
    print(f"  mean_ep_length:   {EL.mean():.1f} (±{EL.std():.1f})")
    print(f"  success_rate:     {SR.mean():.2%}")
    print(f"  obstacle_hits/ep: {HITS.mean():.2f}")
    print(f"  revisits/ep:      {RV.mean():.2f}")
    print(f"  avg_battery_end:  {AB.mean():.2f}")

    return evaluated

# ==============================================================
# Manual control (keyboard) — same shape as evaluate, but user drives
# ==============================================================
def manual_control(
    exp_or_meta,                       # dict metadata OR str experiment folder
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
    is_att = isinstance(arch, str) and arch.startswith("a")

    env_vec = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            arch=arch, temporal_len=temporal_len,
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
