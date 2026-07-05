"""MIKASA-Robo-VLA -> websocket bridge for VLA-JEPA checkpoints.

Mirrors examples/LIBERO/eval_libero.py's client loop against the same model
server (deployment/model_server/server_policy.py):
  - one M1Inference websocket client; reset RPC carries episode_id/task_key/seed
  - the server is queried when step % action_chunk_size == 0 (decision cadence)
  - episodes.jsonl / decisions.jsonl written with the blackout.py record helpers

MIKASA-Robo-VLA specifics (github.com/CognitiveAISystems/MIKASA-Robo, v1.0.0):
  - envs registered by importing mikasa_robo_suite.vla.memory_envs, then wrapped
    with apply_mikasa_vla_wrappers (canonical chain; include_overlays=False)
  - obs["rgb"]: (1, 128, 128, 6) uint8 = base_camera (top-down) ++ hand_camera
    (wrist) on the channel axis -> our 2-camera layout [primary, wrist]
  - obs["proprio"]: (1, 7) float32 = [eef xyz (m), eef rpy (rad), gripper
    opening (m, 0..0.08)]; re-encoded to the LIBERO-style 8-D state
    [xyz, axis-angle(3), +q, -q] when --args.with-state true
  - action: 7-D normalized pd_ee_delta_pose in [-1, 1]:
    [dx, dy, dz, droll, dpitch, dyaw, gripper(+1=open, -1=close)].
    Our head emits [world_vector(3), rotation_delta(3), open_gripper in [0,1]
    (1=open)]; translation/rotation pass through clipped to [-1, 1] and the
    gripper is remapped 1->+1 (open), 0->-1 (close). NOTE: this flips the sign
    vs LIBERO/robosuite, where -1 means open.
  - language instruction from the benchmark manifest (mikasa_robo_vla_envs.csv,
    via mikasa_robo_suite.vla.benchmarking.load_benchmark_tasks)
  - canonical episode seeding: env.reset(seed=start_seed + episode_idx) with
    START_SEED=4242424242; success latched from info["success"] (success_once)

Run inside the maniskill env (memexp_stage/envs/maniskill) with the repo root
on PYTHONPATH; see cluster/eval_mikasa.sbatch.
"""

import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402  (mikasa/mani_skill envs are torch-native)

from examples.LIBERO.blackout import (  # noqa: E402
    checkpoint_sha,
    decision_record,
    episode_record,
    memory_params_from_env,
    read_git_sha,
    resolve_memory_mode,
)
from examples.LIBERO.model2libero_interface import M1Inference  # noqa: E402

import gymnasium as gym  # noqa: E402

# Compat shim: gymnasium>=1.0 removed Wrapper.__getattr__ delegation to the
# inner env. MIKASA v1.0.0 pins gymnasium==0.29.1 and relies on it (e.g.
# FlattenRGBDObservationWrapper reads `self.base_env.device` through a
# StateOnlyTensorToDictWrapper; ppo_memtasks even filters the 0.29 deprecation
# warning for exactly this pattern). Our maniskill env runs gymnasium 1.3, so
# restore the 0.29 behavior. Same spirit as eval_libero.py's torch.load shim.
if not hasattr(gym.Wrapper, "__getattr__"):

    def _wrapper_getattr(self, name: str):
        # Mirror gymnasium 0.29.1: private names never delegate; guard "env"
        # against recursion when a wrapper is not fully initialized.
        if name.startswith("_") or name == "env":
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    gym.Wrapper.__getattr__ = _wrapper_getattr

# Registers the 90 *-VLA-v0 env IDs as an import side effect.
import mikasa_robo_suite.vla.memory_envs  # noqa: F401,E402
from mikasa_robo_suite.vla.benchmarking import (  # noqa: E402
    CONTROL_MODE,
    OBS_MODE,
    REWARD_MODE,
    START_SEED,
    load_benchmark_tasks,
)
from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers  # noqa: E402


def short_name(text: str, max_len: int = 80) -> str:
    """Filesystem-safe task tag (same scheme as eval_libero.py, inlined so this
    module never imports eval_libero, which requires the LIBERO package)."""
    import hashlib

    h = hashlib.md5(text.encode()).hexdigest()[:8]
    clean = text.replace(" ", "_")[:max_len]
    return f"{clean}_{h}"


SUITE_NAME = "mikasa_vla"
# Anchor battery: chance-level for memoryless policies (adoption plan sec 2.2).
DEFAULT_ENV_IDS = "ShellGameTouch-VLA-v0,RememberColor3-VLA-v0,TakeItBack-VLA-v0,ChainOfColors3-VLA-v0"


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]

    #################################################################################################################
    # MIKASA-Robo-VLA environment-specific parameters
    #################################################################################################################
    env_ids: str = DEFAULT_ENV_IDS  # comma-separated *-VLA-v0 env IDs
    num_trials_per_task: int = 50  # canonical NUM_EPISODES_PER_TASK is 50
    start_seed: int = START_SEED  # canonical benchmark seed base (4242424242)
    sim_backend: str = "auto"  # 'auto' == CPU sim + Vulkan GPU render at num_envs=1 (Track B PASS config)
    render_mode: str = "none"  # 'none' keeps the human render camera off (Track B PASS config); 'all' = canonical
    max_steps_cap: int = 0  # 0: use each env's max_episode_steps; >0: hard cap (smoke tests)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/mikasa/logs"  # episodes/decisions jsonl + rollout mp4s
    save_videos: bool = True  # top-camera rollout mp4 per episode
    pretrained_path: str = ""
    with_state: str = "true"
    # Multi-embodiment checkpoints: MIKASA's arm is a Panda -> Franka stats block.
    unnorm_key: str = "franka"
    job_name: str = "test"


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rpy_to_axisangle(rpy: np.ndarray) -> np.ndarray:
    """XYZ-Euler (roll, pitch, yaw) -> axis-angle vector (LIBERO state rotation encoding)."""
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", rpy).as_rotvec().astype(np.float32)


def _state_from_proprio(proprio: np.ndarray) -> np.ndarray:
    """MIKASA 7-D proprio -> LIBERO-style 8-D state.

    [x, y, z, axis-angle(3), grip/2, -grip/2]: the trailing pair mimics
    robosuite's two-finger gripper_qpos (opening in [0, 0.08] m split
    symmetrically, second finger mirrored negative).
    """
    p = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if p.size != 7:
        raise ValueError(f"expected 7-D MIKASA proprio, got shape {p.shape}")
    axangle = _rpy_to_axisangle(p[3:6])
    half = 0.5 * p[6]
    return np.concatenate([p[:3], axangle, [half, -half]]).astype(np.float32)


def _mikasa_action(raw_action: dict) -> np.ndarray:
    """Model raw_action -> 7-D normalized pd_ee_delta_pose action."""
    world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
    if not (world_vector.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
        raise ValueError(
            f"Invalid action sizes: world_vector={world_vector.shape}, "
            f"rotation_delta={rotation_delta.shape}, gripper={open_gripper.shape}"
        )
    # ManiSkill Panda gripper: +1 opens, -1 closes (opposite sign of robosuite).
    gripper = np.asarray([2.0 * (float(open_gripper[0]) > 0.5) - 1.0], dtype=np.float32)
    action = np.concatenate([world_vector, rotation_delta, gripper])
    return np.clip(action, -1.0, 1.0)


def _first_scalar(value, default):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.detach().reshape(-1)[0].cpu().item()
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _make_env(env_id: str, sim_backend: str, render_mode: str):
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode=OBS_MODE,  # "rgb"
        control_mode=CONTROL_MODE,  # "pd_ee_delta_pose"
        reward_mode=REWARD_MODE,  # "normalized_dense"
        render_mode=None if render_mode == "none" else render_mode,
        sim_backend=sim_backend,
    )
    return apply_mikasa_vla_wrappers(env, include_overlays=False)


def eval_mikasa(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.start_seed % (2**32))

    env_ids = [x.strip() for x in args.env_ids.split(",") if x.strip()]
    all_tasks = load_benchmark_tasks()
    manifest = {t.env_id: t for t in all_tasks}
    # BenchmarkTask drops the CSV id column; recover it from the (stable) row order.
    manifest_ids = {t.env_id: i for i, t in enumerate(all_tasks)}
    unknown = [e for e in env_ids if e not in manifest]
    if unknown:
        raise ValueError(f"env_ids not in mikasa_robo_vla_envs.csv manifest: {unknown}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Self-describing episode records (same protocol as eval_libero.py): the
    # server owns the memory policy; env MEMORY_MODE is a fallback/cross-check.
    env_memory_mode = os.environ.get("MEMORY_MODE")
    memory_params = memory_params_from_env()
    git_sha = read_git_sha(pathlib.Path(__file__).resolve().parent)
    ckpt_sha = checkpoint_sha(args.pretrained_path)
    episodes_path = pathlib.Path(args.video_out_path) / "episodes.jsonl"
    decisions_path = pathlib.Path(args.video_out_path) / "decisions.jsonl"
    for record_path in (episodes_path, decisions_path):
        record_path.write_text("")

    model = M1Inference(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        unnorm_key=args.unnorm_key,
    )

    total_episodes, total_successes = 0, 0
    for env_id in tqdm.tqdm(env_ids):
        task = manifest[env_id]
        task_description = task.language_instruction
        env = _make_env(env_id, args.sim_backend, args.render_mode)
        max_steps = int(getattr(env, "max_episode_steps"))
        if args.max_steps_cap > 0:
            max_steps = min(max_steps, args.max_steps_cap)
        logging.info(f"Task {env_id} ({task.memory_type}/{task.split}): '{task_description}', max_steps={max_steps}")

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            model.reset(
                task_description=task_description,
                episode_id=f"{SUITE_NAME}--{env_id}--ep{episode_idx}",
                task_key=f"{SUITE_NAME}--{env_id}",
            )
            episode_seed = model._episode_counter  # seed sent with the reset RPC
            env_seed = args.start_seed + episode_idx  # canonical benchmark env seed
            obs, _ = env.reset(seed=env_seed)

            step = 0
            num_decisions = 0
            success_once = False
            decision_rows = []
            first_memory_extras = None
            replay_images = []

            while step < max_steps:
                rgb = _to_numpy(obs["rgb"])[0].astype(np.uint8)  # (128, 128, 6)
                img = np.ascontiguousarray(rgb[:, :, :3])  # base_camera (top-down) = primary
                wrist_img = np.ascontiguousarray(rgb[:, :, 3:6])  # hand_camera = wrist
                if args.save_videos:
                    replay_images.append(img)

                obs_input = {
                    "images": [img, wrist_img],
                    "task_description": task_description,
                    "step": step,
                }
                if args.with_state == "true":
                    proprio = _to_numpy(obs["proprio"])[0]
                    obs_input["state"] = _state_from_proprio(proprio)[None]

                decision_idx = step // model.action_chunk_size
                start_time = time.time()
                response = model.step(**obs_input)
                infer_secs = time.time() - start_time

                if step % model.action_chunk_size == 0:
                    num_decisions += 1
                    if first_memory_extras is None:
                        first_memory_extras = model.last_memory_extras
                    decision_rows.append(
                        decision_record(
                            episode_idx=episode_idx,
                            d=decision_idx,
                            memory_extras=model.last_memory_extras,
                            blackout_active=False,
                            extras={"env_id": env_id, "infer_secs": round(infer_secs, 4)},
                        )
                    )

                delta_action = _mikasa_action(response["raw_action"])
                action = torch.as_tensor(delta_action[None], dtype=torch.float32)
                device = getattr(env.unwrapped, "device", None)
                if device is not None:
                    action = action.to(device)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                success_once = success_once or bool(_first_scalar(info.get("success"), default=False))
                done = bool(_first_scalar(terminated, default=False)) or bool(
                    _first_scalar(truncated, default=False)
                )
                if done:
                    break

            task_episodes += 1
            total_episodes += 1
            if success_once:
                task_successes += 1
                total_successes += 1

            if args.save_videos and replay_images:
                suffix = "success" if success_once else "failure"
                task_segment = short_name(env_id)
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            logging.info(f"Success(once): {success_once}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

            server_extras = first_memory_extras or {}
            record = episode_record(
                suite=SUITE_NAME,
                task_id=manifest_ids[env_id],
                task_description=task_description,
                episode_idx=episode_idx,
                memory_mode=resolve_memory_mode(env_memory_mode, server_extras.get("mode")),
                memory_params=memory_params,
                episode_seed=episode_seed,
                success=success_once,
                num_env_steps=step,
                num_decisions=num_decisions,
                ckpt=args.pretrained_path,
                ckpt_sha=ckpt_sha,
                git_sha=git_sha,
                extras={
                    "env_id": env_id,
                    "env_seed": env_seed,
                    "memory_type": task.memory_type,
                    "horizon_split": task.split,
                    "max_steps": max_steps,
                    "donor_source": server_extras.get("donor_episode"),
                },
            )
            with open(episodes_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            if decision_rows:
                with open(decisions_path, "a") as f:
                    f.writelines(json.dumps(row) + "\n" for row in decision_rows)

        env.close()
        logging.info(
            f"[{env_id}] task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_mikasa)
