# train.py

import os
import traceback
import time
import hashlib
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from src.MineEnv import MineEnv
from src.constants import SAVE_DIR
from src.utils import *  # make_experiment_name, get_metadata, etc.
from src.wrappers import TimeStackObservation
from src.attention import AttentionCNNExtractor
from src.cnn_feature_extractor import GridCNNExtractor
from src.reward_functions import *  # includes reward_d, compute_reward

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

    # ---- choose extractor + kwargs ----
    if is_att:
        policy_kwargs = dict(
            features_extractor_class=AttentionCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                temporal_len=int(temporal_len),
            ),
        )
    elif is_cnn:
        policy_kwargs = dict(
            features_extractor_class=GridCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
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

    # Persist action order once for reproducibility/debug (if available)
    try:
        actions_order = env.get_attr("_ACTIONS")[0]
        with open(os.path.join(tb_root, "actions_order.txt"), "w") as f:
            f.write(repr(actions_order))
        print("[TRAIN] actions order:", actions_order)
    except Exception:
        pass

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
    model_path = save(model, run_dir, suffix="")
    return model_path, run_dir

# ===== Tiny eval helpers =====

def _desc_space(space):
    try:
        cls = space.__class__.__name__
        shp = getattr(space, "shape", None)
        dtype = getattr(space, "dtype", None)
        return f"{cls}(shape={tuple(shp)}, dtype={dtype})" if shp is not None else cls
    except Exception as e:
        return f"<unprintable space: {e}>"


def _peek_policy(env, model, steps=5, deterministic=True):
    """
    Print argmax action, entropy, and probs for a few steps.
    """
    obs = env.reset()
    raw_env = env.envs[0]

    for i in range(steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=model.device)
            dist = model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            action = int(probs.argmax())
            p = np.clip(probs, 1e-8, 1.0)
            entropy = float(-(p * np.log(p)).sum())
        print(
            f"[POLICY {i}] argmax={action}  "
            f"entropy={entropy:.3f}  probs={np.array2string(probs, precision=3, suppress_small=True)}"
        )
        obs, _, dones, _ = env.step([action] if deterministic else [raw_env.action_space.sample()])
        if np.asarray(dones).any():
            obs = env.reset()

# ---- Deep-dive debug helpers (optional) ----

def _hash(a):
    try:
        return hashlib.sha1(np.ascontiguousarray(a).view(np.uint8)).hexdigest()[:10]
    except Exception:
        return "na"


def _summarize_obs(obs, tag=""):
    arr = np.asarray(obs)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4:
        T, C, H, W = arr.shape
        t_std = arr.reshape(T, -1).std(axis=1)
        ch_std = arr.std(axis=(0, 2, 3))
        same_as_t0 = [bool(np.allclose(arr[t], arr[0])) for t in range(T)]
        frame_hashes = [_hash(arr[t]) for t in range(T)]
        ch_nnz = arr.astype(bool).sum(axis=(0, 2, 3))
        print(f"[OBS{tag}] shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")
        print(f"[OBS{tag}] per-time std: {np.array2string(t_std, precision=4, threshold=24)}")
        print(f"[OBS{tag}] per-channel std: {np.array2string(ch_std, precision=4, threshold=24)}")
        print(f"[OBS{tag}] per-channel nnz: {np.array2string(ch_nnz, max_line_width=200)}")
        print(f"[OBS{tag}] frames_equal_to_t0: {same_as_t0}")
        print(f"[OBS{tag}] frame_hashes: {frame_hashes}")
        if all(same_as_t0):
            print("[OBS] ⚠ time stack appears constant within this observation.")
    elif arr.ndim == 3:
        C, H, W = arr.shape
        ch_std = arr.std(axis=(1, 2))
        ch_nnz = arr.astype(bool).sum(axis=(1, 2))
        print(f"[OBS{tag}] shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")
        print(f"[OBS{tag}] per-channel std: {np.array2string(ch_std, precision=4, threshold=24)}")
        print(f"[OBS{tag}] per-channel nnz: {np.array2string(ch_nnz, max_line_width=200)}")
    else:
        print(f"[OBS{tag}] unexpected ndim={arr.ndim} shape={arr.shape}")


def _peek_logits_probs_value(model, obs, tag=""):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, device=model.device)
        dist = model.policy.get_distribution(obs_t)
        logits = dist.distribution.logits.detach().cpu().numpy()[0]
        probs  = dist.distribution.probs.detach().cpu().numpy()[0]
        p = np.clip(probs, 1e-8, 1.0)
        entropy = float(-(p * np.log(p)).sum())
        try:
            v = float(model.policy.predict_values(obs_t).detach().cpu().numpy()[0])
        except Exception:
            v = None
    print(f"[POLICY{tag}] logits={np.array2string(logits, precision=3, suppress_small=True)}")
    print(f"[POLICY{tag}] probs ={np.array2string(probs,  precision=3, suppress_small=True)}  argmax={int(probs.argmax())}  entropy={entropy:.3f}  V={v}")


# ---------- NEW: unwrap helpers so probes can reach MineEnv through wrappers ----------

def _unwrap_env(env):
    """
    Recursively unwrap common Gym wrappers to reach the base environment
    (your MineEnv). Returns (base_env, wrapper_chain_names).
    """
    chain = [env.__class__.__name__]
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
        chain.append(cur.__class__.__name__)
    return cur, chain


def _wrapper_stack_str(chain):
    try:
        return " -> ".join(chain)
    except Exception:
        return str(chain)


def _print_reward_landscape(env_like):
    """
    Print immediate reward for each legal neighbor move from current pos.
    Works whether you pass Monitor, TimeStackObservation, or MineEnv.
    """
    base_env, _ = _unwrap_env(env_like)
    try:
        acts = getattr(base_env, "_ACTIONS")
    except Exception:
        acts = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]

    sim = getattr(base_env, "simulator", None)
    if sim is None or not hasattr(base_env, "reward_fn"):
        print("[LANDSCAPE] base env missing .simulator or .reward_fn")
        return

    r0, c0 = tuple(sim.guided_miner_pos)
    print("[LANDSCAPE] pos=", (r0, c0))
    rows = []
    for i, (dr, dc) in enumerate(acts):
        new_pos = (r0 + dr, c0 + dc)
        rew, sub = compute_reward(base_env, base_env.reward_fn, new_pos=new_pos)
        rows.append((i, (dr, dc), float(rew), sub))
    rows.sort(key=lambda t: t[2], reverse=True)
    for i, move, rew, sub in rows:
        try:
            dpen = sub.get("distance_penalty", None)
        except Exception:
            dpen = None
        print(f"  a={i:<2} move={move!s:<9} reward={rew:>7.3f}  distance_penalty={dpen}")


def _print_local_neighborhood(env_like):
    """
    Print legality, path hints, battery-tier for each neighbor.
    Works through wrappers by unwrapping first.
    """
    base_env, _ = _unwrap_env(env_like)
    sim = getattr(base_env, "simulator", None)
    if sim is None:
        print("[LOCAL] base env missing .simulator")
        return

    H, W = sim.n_rows, sim.n_cols
    try:
        acts = getattr(base_env, "_ACTIONS")
    except Exception:
        acts = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    r0, c0 = tuple(sim.guided_miner_pos)

    def _cheb(a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    goals = sim.goal_positions or [(r0, c0)]
    d0 = min(_cheb((r0, c0), g) for g in goals)

    try:
        path_rc = {(y, x) for (x, y) in base_env.pathfinder.getShortestPath()} if base_env.pathfinder else set()
    except Exception:
        path_rc = set()

    print(f"[LOCAL] pos={(r0,c0)}  d*len={len(path_rc)}  batt_map={'ok' if getattr(base_env,'batt_map',None) is not None else 'None'}")

    def _safe(mat, r, c, default=0.0):
        H2, W2 = mat.shape
        if 0 <= r < H2 and 0 <= c < W2:
            return float(mat[r, c])
        return default

    try:
        from src.constants import COST_TABLE
    except Exception:
        COST_TABLE = {}

    for i, (dr, dc) in enumerate(acts):
        rr, cc = r0 + dr, c0 + dc
        inb = (0 <= rr < H and 0 <= cc < W)
        free = bool(inb and sim.is_valid_guided_miner_move((rr, cc)))
        on_path = (rr, cc) in path_rc
        batt = _safe(base_env.batt_map, rr, cc, default=0.0) if (getattr(base_env, 'batt_map', None) is not None and inb) else 0.0
        try:
            cost_tier = COST_TABLE[int(batt)]
        except Exception:
            cost_tier = None
        d_new = min(_cheb((rr, cc), g) for g in goals) if inb else None
        print(f"  a={i} move=({dr:+d},{dc:+d}) inb={inb} free={free} on_d*={on_path} batt={batt:5.1f} cost={cost_tier} d_old={d0} d_new={d_new}")


# ===== Auto-discover a single model zip under the experiment folder =====

def _auto_model_path(experiment_folder, arch):
    """
    Search typical roots for model*.zip under the experiment folder.
    Preference bucket order:
      1) model_{arch}.zip (if arch != 'mlp')
      2) model.zip
      3) any other model*.zip
    Within a bucket: prefer higher PPO run number, then newer mtime.
    """
    roots = []
    try:
        roots.append(SAVE_DIR)  # configured project save dir
    except NameError:
        pass
    roots.extend([os.path.join(os.getcwd(), "saved_experiments"), os.getcwd()])
    roots = [r for r in dict.fromkeys(roots) if os.path.isdir(r)]

    candidates = []  # (bucket, run_num, mtime, path)
    preferred1 = f"model_{arch}.zip" if arch and arch != "mlp" else None

    for root in roots:
        exp_dir = os.path.join(root, experiment_folder)
        if not os.path.isdir(exp_dir):
            continue
        for dirpath, _, files in os.walk(exp_dir):
            for fn in files:
                if not (fn.startswith("model") and fn.endswith(".zip")):
                    continue
                path = os.path.join(dirpath, fn)
                # bucket
                if preferred1 and fn == preferred1:
                    bucket = 0
                elif fn == "model.zip":
                    bucket = 1
                else:
                    bucket = 2
                # PPO run number if parent dir is PPO_<n>
                parent = os.path.basename(os.path.dirname(path))
                try:
                    run_num = int(parent.split("_", 1)[1]) if parent.startswith("PPO_") else -1
                except Exception:
                    run_num = -1
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    mtime = 0.0
                candidates.append((bucket, run_num, mtime, path))

    if not candidates:
        raise FileNotFoundError(
            f"No model*.zip found under any of: {roots}\n"
            f"Expected something like .../<experiment>/PPO_x/model*.zip"
        )

    # Choose best bucket, then sort within it: run_num desc, mtime desc
    best_bucket = min(c[0] for c in candidates)
    bucketed = [c for c in candidates if c[0] == best_bucket]
    bucketed.sort(key=lambda t: (t[1], t[2]), reverse=True)
    model_path = bucketed[0][3]
    print(f"[MODEL] selected: {model_path}")
    return model_path


# ===== Evaluate (auto model; path optional) =====

def evaluate(
    exp_or_meta,                       # dict metadata OR str experiment folder
    mode: str = "static",
    use_planner_overlay: bool = True,
    show_miners: bool = False,
    show_predicted: bool = True,
    total_timesteps: int = 300,
    render: bool = False,
    temporal_len: int = 12,
    policy_debug_steps: int = 4,       # >0 to dump policy stats a few steps
    deterministic: bool = True,        # greedy policy by default
    load_model_path: str = None,       # OPTIONAL: honored if provided
    verbose: bool = False,             # print per-timestep info if True
    debug_probe_steps: int = 4,        # if >0, run deep-dive probes for first N steps
):
    """
    Evaluate a trained PPO model.

    You do NOT need to pass a model path. If omitted, we auto-discover one
    under the experiment folder (prefers model_{arch}.zip, then model.zip).
    """
    # --- Normalize input via project helpers ---
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

    # --- Choose checkpoint ---
    if load_model_path:
        model_path = load_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"load_model_path does not exist: {model_path}")
    else:
        model_path = _auto_model_path(experiment_folder, arch)

    run_dir = os.path.dirname(model_path)
    run_name = os.path.basename(run_dir)

    # --- Build eval env (arch-aware) ---
    eval_env = DummyVecEnv([
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

    try:
        # Grab both the wrapper and the base env
        raw_wrapper = eval_env.envs[0]          # Monitor(...)
        base_env, chain = _unwrap_env(raw_wrapper)

        # Spaces + mapping introspection
        print(f"[DEBUG] arch='{arch}' | is_cnn={is_cnn} | is_att={is_att} | temporal_len={temporal_len}")
        print(f"[DEBUG] VecEnv observation_space: {_desc_space(eval_env.observation_space)}")
        print(f"[DEBUG] VecEnv action_space:      {_desc_space(eval_env.action_space)}")
        print(f"[DEBUG] env wrapper stack: {_wrapper_stack_str(chain)}")
        try:
            print(f"[DEBUG] Raw wrapper observation_space: {_desc_space(raw_wrapper.observation_space)}")
            print(f"[DEBUG] Raw wrapper action_space:      {_desc_space(raw_wrapper.action_space)}")
            print(f"[DEBUG] Base env observation_space:    {_desc_space(base_env.observation_space)}")
            print(f"[DEBUG] Base env action_space:         {_desc_space(base_env.action_space)}")
            if hasattr(raw_wrapper, "_has_time_stack"):
                print(f"[DEBUG] _has_time_stack={raw_wrapper._has_time_stack}  "
                      f"_tstack_len={getattr(raw_wrapper, '_tstack_len', None)}")
            for name in ("_ACTIONS", "ACTION_TO_MOVE_MAP", "MOVE_TO_ACTION_MAP"):
                if hasattr(base_env, name):
                    print(f"[DEBUG] base.{name}: {getattr(base_env, name)}")
                elif hasattr(raw_wrapper, name):
                    print(f"[DEBUG] wrap.{name}: {getattr(raw_wrapper, name)}")
        except Exception as e:
            print(f"[WARN] could not introspect env mappings: {e}")

        # --- Load model ---
        print(f"[INFO] Loading {model_path} (run {run_name})")
        model = PPO.load(model_path, env=eval_env, device="auto", print_system_info=False)

        # Confirm loaded spaces + extractor
        try:
            print(f"[DEBUG] Loaded model observation_space: "
                  f"{_desc_space(getattr(model, 'observation_space', None))}")
            print(f"[DEBUG] Loaded model action_space:      "
                  f"{_desc_space(getattr(model, 'action_space', None))}")
            fx = getattr(getattr(model, "policy", None), "features_extractor", None)
            if fx is not None:
                print(f"[DEBUG] features_extractor: {fx.__class__.__name__}")
        except Exception:
            pass

        # --- Optional quick policy peek ---
        if policy_debug_steps and policy_debug_steps > 0:
            _peek_policy(eval_env, model, steps=policy_debug_steps, deterministic=deterministic)

        if total_timesteps <= 0 and not debug_probe_steps:
            return  # inspection-only

        # Pre-fetch action list for pretty printing (if available)
        try:
            actions_list = eval_env.get_attr("_ACTIONS")[0]
        except Exception:
            actions_list = None

        # One-time deep probes at t=0
        obs = eval_env.reset()
        if debug_probe_steps and debug_probe_steps > 0:
            try:
                _summarize_obs(obs, tag=" t=0")                      # time stack stats
                _print_reward_landscape(base_env)                      # local reward-by-move
                _print_local_neighborhood(base_env)                    # neighborhood facts
                _peek_logits_probs_value(model, obs, tag=" t=0")       # logits/probs/value
            except Exception as e:
                print(f"[debug_probe] error: {e}")

        # --- Main eval loop ---
        steps = 0
        while steps < total_timesteps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = eval_env.step(action)
            steps += 1

            if verbose:
                try:
                    a_idx = int(action[0]) if hasattr(action, "__len__") else int(action)
                except Exception:
                    a_idx = int(action)
                info0 = infos[0] if isinstance(infos, (list, tuple)) and infos else (infos or {})
                r = rewards[0] if isinstance(rewards, (list, tuple, np.ndarray)) else rewards
                bits = [f"t={steps}", f"action={a_idx}"]
                if actions_list and 0 <= a_idx < len(actions_list):
                    bits.append(f"move={actions_list[a_idx]}")
                bits.append(f"reward={_fmt(r)}")
                for k in ("current_battery", "distance_to_goal"):
                    if k in info0 and info0[k] is not None:
                        bits.append(f"{k}={_fmt(info0[k])}")
                subs = info0.get("subrewards")
                if isinstance(subs, dict) and subs:
                    sub_str = ", ".join(f"{k}={_fmt(v)}" for k, v in subs.items())
                    bits.append(f"subs[{sub_str}]")
                print(" | ".join(bits))

            if debug_probe_steps and steps <= debug_probe_steps:
                try:
                    _peek_logits_probs_value(model, obs, tag=f" t={steps}")
                    _print_local_neighborhood(base_env)
                except Exception as e:
                    print(f"[debug_probe step] error: {e}")

            try:
                if np.asarray(dones).any():
                    obs = eval_env.reset()
            except Exception:
                pass

    finally:
        try:
            eval_env.close()
        except Exception:
            pass

# ==============================================================
# Testing / batch training helpers
# ==============================================================

def train_all(total_timesteps: int = 1_000_000):
    """
    Define a list of experiments and train each. By default trains for 1,000,000 steps.
    """
    experiments = [
        {"grid": "mine_50x50", "miners": 20, "arch": "a0", "reward": "reward_d"},
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
    ppo_runs = []
    for name in os.listdir(exp_dir):
        full = os.path.join(exp_dir, name)
        if os.path.isdir(full) and name.startswith("PPO_"):
            ppo_runs.append((name, full))
    if ppo_runs:
        def _k(t):
            n = t[0].split("_", 1)[-1]
            return int(n) if n.isdigit() else 0
        ppo_runs.sort(key=_k)
        for _, run_dir in reversed(ppo_runs):
            zips = [f for f in os.listdir(run_dir) if f.startswith("model") and f.endswith(".zip")]
            if zips:
                zips.sort()
                return run_dir, os.path.join(run_dir, zips[-1])

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
                    load_model_path=model_zip,     # pick THIS run's model
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
                try:
                    env.close()
                except Exception:
                    pass
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
                    er = [m["episode_return"] for m in ep_metrics]
                    el = [m["episode_length"] for m in ep_metrics]
                    sr = [1.0 if m["reached_goal"] else 0.0 for m in ep_metrics]
                    print(
                        f"  [run summary] episodes={len(ep_metrics)} | "
                        f"mean_return={np.mean(er):.3f} | mean_len={np.mean(el):.1f} | "
                        f"success_rate={np.mean(sr):.2%}"
                    )
                else:
                    print("  [run summary] no completed episodes")

            except KeyboardInterrupt:
                print("  [interrupted]")
            except Exception as e:
                print(f"  [error during eval] {e}")
            finally:
                try:
                    env.close()
                except Exception:
                    pass

            evaluated.append((exp_name, run_dir, model_zip))

    if not all_episode_metrics:
        print("\n[evaluate_all] No completed episodes across all runs.")
        return evaluated

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

    env_vec = DummyVecEnv([
        _make_env_thunk(
            experiment_folder, mode, use_planner_overlay,
            show_miners, show_predicted, render=render,
            arch=arch, temporal_len=temporal_len,
            reward_fn=rfn,
        )
    ])

    try:
        import pygame
    except Exception as e:
        print(f"[manual_control] pygame unavailable: {e}")
        env_vec.close()
        return

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
        obs = env_vec.reset()

        try:
            actions_list = env_vec.get_attr("_ACTIONS")[0]
        except Exception:
            actions_list = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        drc_to_idx = {drc: i for i, drc in enumerate(actions_list)}

        clock = pygame.time.Clock()
        steps = 0

        def _poll_action_index():
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

        while steps < total_timesteps:
            act_idx = _poll_action_index()
            if act_idx == -1:
                break
            if act_idx is None:
                time.sleep(0.01)
                clock.tick(max(1, int(fps)))
                continue

            action = np.array([act_idx], dtype=np.int64)
            obs, rewards, dones, infos = env_vec.step(action)
            steps += 1

            _log_step(steps, rewards, infos)

            if dones[0]:
                obs = env_vec.reset()

            clock.tick(max(1, int(fps)))

    finally:
        env_vec.close()
