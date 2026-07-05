"""RoboMME (arXiv 2603.04639, ICML 2026 Spotlight) -> VLA-JEPA websocket bridge.

Same pattern as examples/LIBERO/eval_libero.py: the model server (VLA_JEPA env,
deployment/model_server/server_policy.py) owns the memory policy; this client
runs the simulator (RoboMME/ManiSkill3, in the robomme uv venv at
memexp_stage/robomme/.venv), streams 2-view RGB + state over the websocket, and
writes episodes.jsonl / decisions.jsonl with the MEMORY_MODE cross-check.

RoboMME specifics handled here:
  - Their obs are LISTS (conditioning-video tasks return demo frames on reset).
    The demo frames are optionally streamed through the model (actions
    discarded) so the recurrent memory ingests the cue: --args.ingest-conditioning.
  - Their `ee_pose` action space is ABSOLUTE world-frame [x,y,z,r,p,y,gripper]
    with rpy unwrapped extrinsic-XYZ and gripper -1=close/+1=open. Our
    LIBERO-trained checkpoints emit 7-DoF DELTA-EE with gripper -1=open/+1=close,
    so `--args.action-mode delta_to_abs` (default) integrates deltas onto the
    current eef_state with --args.pos-scale/--args.rot-scale (robosuite OSC
    output_max heuristics) and flips the gripper sign. `absolute` passes the
    unnormalized model output straight through (for future checkpoints
    fine-tuned on robomme_to_lerobot.py data with a `robomme` unnorm_key).
  - Benchmark protocol: fixed-seed splits train=100 / val=50 / test=50 episodes
    per task; official MME-VLA eval = all 50 test episodes, max_steps=1300.

No starVLA import (the sim venv doesn't carry it): norm stats + action chunk
size are read directly from the run dir next to the checkpoint
(dataset_statistics.json + config.yaml), matching M1Inference semantics.
"""

import dataclasses
import json
import logging
import os
import pathlib
import time

import cv2
import imageio
import numpy as np
import tqdm
import tyro
import yaml

# Registers the 16 RoboMME env ids with ManiSkill's gym registry.
import robomme.robomme_env  # noqa: F401
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.LIBERO.blackout import (
    checkpoint_sha,
    decision_record,
    episode_record,
    memory_params_from_env,
    read_git_sha,
    resolve_memory_mode,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUITE_OF = {
    "BinFill": "Counting", "PickXtimes": "Counting", "SwingXtimes": "Counting",
    "StopCube": "Counting",
    "VideoUnmask": "Permanence", "VideoUnmaskSwap": "Permanence",
    "ButtonUnmask": "Permanence", "ButtonUnmaskSwap": "Permanence",
    "PickHighlight": "Reference", "VideoRepick": "Reference",
    "VideoPlaceButton": "Reference", "VideoPlaceOrder": "Reference",
    "MoveCube": "Imitation", "InsertPeg": "Imitation",
    "PatternLock": "Imitation", "RouteStick": "Imitation",
}
TASK_INDEX = {name: i for i, name in enumerate(BenchmarkEnvBuilder.get_task_list())}
EPISODE_LIMITS = {"train": 100, "val": 50, "test": 50}
# Stick-manipulation tasks take 6-D ee_pose actions (the wrapper rejects a
# gripper channel: EndeffectorDemonstrationWrapper._EE_POSE_7D_ENV_IDS).
NO_GRIPPER_TASKS = ("PatternLock", "RouteStick")


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _frame(x) -> np.ndarray:
    """RoboMME RGB list entry -> (H, W, 3) uint8 (drops a possible batch dim)."""
    arr = _to_numpy(x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return np.ascontiguousarray(arr.astype(np.uint8))


def _vec(x, n: int) -> np.ndarray:
    arr = _to_numpy(x).astype(np.float32).reshape(-1)
    if arr.size < n:
        raise ValueError(f"expected >= {n} dims, got shape {arr.shape}")
    return arr[:n]


def _rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic-XYZ (static) roll/pitch/yaw -> axis-angle 3-vector."""
    from transforms3d.euler import euler2axangle
    axis, angle = euler2axangle(float(rpy[0]), float(rpy[1]), float(rpy[2]), axes="sxyz")
    return np.asarray(axis, dtype=np.float32) * np.float32(angle)


def _build_state(obs: dict, idx: int = -1) -> np.ndarray:
    """8-D state in the franka training convention: [xyz, axis-angle, gripper(2)].

    ``idx`` selects the list entry (default: latest); conditioning-video frames
    are paired with their own recorded states.
    """
    idx = min(idx, len(obs["eef_state_list"]) - 1) if idx >= 0 else idx
    eef = _vec(obs["eef_state_list"][idx], 6)
    grip = _vec(obs["gripper_state_list"][idx], 2)
    return np.concatenate([eef[:3], _rpy_to_axis_angle(eef[3:6]), grip]).astype(np.float32)


class RoboMMEPolicyClient:
    """Chunked websocket policy client (M1Inference minus the starVLA imports).

    The action chunk size and unnorm stats are read from the checkpoint's run
    dir (config.yaml / dataset_statistics.json), so behavior matches
    examples/LIBERO/model2libero_interface.M1Inference exactly: one server
    infer per chunk of `future_action_window_size + 1` env steps, [-1, 1]
    clipping, min/max unnormalization under the mask, and a 0/1 binarized
    gripper channel at index 6.
    """

    def __init__(self, ckpt_path: str, host: str, port: int,
                 unnorm_key: str = "franka", image_size=(224, 224)) -> None:
        # Literal path (no symlink resolution) to match starVLA read_mode_config:
        # exported live/zero ckpts are symlinked identical weights whose run dirs differ.
        run_dir = pathlib.Path(ckpt_path).absolute().parents[1]
        stats_path = run_dir / "dataset_statistics.json"
        config_path = run_dir / "config.yaml"
        norm_stats = json.loads(stats_path.read_text())
        if unnorm_key not in norm_stats:
            raise KeyError(f"unnorm_key {unnorm_key!r} not in {sorted(norm_stats)} ({stats_path})")
        self.action_norm_stats = norm_stats[unnorm_key]["action"]
        cfg = yaml.safe_load(config_path.read_text())
        self.action_chunk_size = int(cfg["framework"]["action_model"]["future_action_window_size"]) + 1

        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.task_description = None
        self.last_memory_extras = None
        self.raw_actions = None
        self._episode_counter = 0

    def reset(self, task_description: str, *, episode_id: str, task_key: str) -> None:
        self._episode_counter += 1
        self.client.reset(
            instruction=task_description,
            episode_id=episode_id,
            episode_seed=self._episode_counter,
            task_key=task_key,
        )
        self.task_description = task_description
        self.last_memory_extras = None
        self.raw_actions = None

    @property
    def episode_seed(self) -> int:
        return self._episode_counter

    def _resize(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def _infer_once(self, images, state, suppress_write: bool = False) -> np.ndarray:
        vla_input = {
            "batch_images": [[self._resize(img) for img in images]],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": 10,
        }
        if state is not None:
            # (1, 8) inside the batch list, matching eval_libero's
            # np.expand_dims(state, 0): the action head concatenates
            # state_features with 3-D token tensors.
            vla_input["state"] = [np.asarray(state, dtype=np.float32)[None, :]]
        if suppress_write:
            vla_input["suppress_write"] = True
        response = self.client.infer(vla_input)
        if "data" not in response:
            raise RuntimeError(f"server returned no data (error?): {response}")
        self.last_memory_extras = response["data"].get("memory_extras")
        normalized = np.asarray(response["data"]["normalized_actions"])[0]
        return self._unnormalize(normalized)

    def ingest(self, images, state) -> None:
        """Feed a conditioning-video frame through the model; discard actions."""
        self._infer_once(images, state)

    def step(self, images, state, step: int) -> np.ndarray:
        """Raw (unnormalized) 7-D action for this env step; server hit on chunk boundaries."""
        if step % self.action_chunk_size == 0:
            self.raw_actions = self._infer_once(images, state)
        return self.raw_actions[step % self.action_chunk_size]

    def _unnormalize(self, normalized_actions: np.ndarray) -> np.ndarray:
        stats = self.action_norm_stats
        mask = np.asarray(stats.get("mask", np.ones_like(stats["min"], dtype=bool)))
        high, low = np.asarray(stats["max"]), np.asarray(stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        return np.where(
            mask, 0.5 * (normalized_actions + 1) * (high - low) + low, normalized_actions
        )


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: tuple[int, int] = (224, 224)

    # ---------------- RoboMME benchmark parameters ----------------
    tasks: str = "PickHighlight"  # comma-separated RoboMME task ids, or "all"
    dataset: str = "test"  # official protocol: fixed seeds, train=100 / val=50 / test=50 eps
    num_episodes: int = 50  # episodes per task (<= split size); official test protocol = 50
    episode_start: int = 0  # first episode index (for sharding / smokes)
    max_steps: int = 1300  # env-step cap; 1300 matches the MME-VLA experiments

    # ---------------- action-space bridging ----------------
    action_mode: str = "delta_to_abs"  # "delta_to_abs" (LIBERO-trained ckpts) | "absolute"
    # Delta scales: robosuite OSC output_max would be 0.05/0.5, but RoboMME's
    # ee_pose wrapper solves IK to 1 mm tolerance and errors the episode when a
    # step target jumps too far — empirically scales this size keep IK alive.
    pos_scale: float = 0.02  # metres per unit delta
    rot_scale: float = 0.1  # radians per unit delta

    # ---------------- conditioning video ----------------
    ingest_conditioning: bool = True  # stream demo frames through the model (memory writes)
    conditioning_stride: int = 0  # frames between ingests; 0 = action chunk size

    # ---------------- utils ----------------
    video_out_path: str = "experiments/robomme/logs"
    seed: int = 7
    pretrained_path: str = ""
    with_state: str = "true"
    unnorm_key: str = "franka"


def resolve_tasks(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return BenchmarkEnvBuilder.get_task_list()
    tasks = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_INDEX]
    if unknown:
        raise ValueError(f"unknown RoboMME tasks {unknown}; valid: {sorted(TASK_INDEX)}")
    return tasks


def to_robomme_action(raw: np.ndarray, eef_state: np.ndarray, args: Args,
                      no_gripper: bool = False) -> np.ndarray:
    """Model output -> RoboMME ee_pose action [x,y,z,r,p,y(,gripper)]."""
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.size != 7:
        raise ValueError(f"expected 7-D action, got {raw.shape}")
    # Model gripper channel is 0/1 after unnormalization (1 = open);
    # RoboMME wants +1 = open, -1 = close (opposite sign of LIBERO's env input).
    gripper = [] if no_gripper else [np.float32(1.0 if raw[6] > 0.5 else -1.0)]
    if args.action_mode == "absolute":
        return np.concatenate([raw[:6], gripper]).astype(np.float32)
    if args.action_mode == "delta_to_abs":
        pos = eef_state[:3] + raw[:3] * args.pos_scale
        rpy = eef_state[3:6] + raw[3:6] * args.rot_scale
        return np.concatenate([pos, rpy, gripper]).astype(np.float32)
    raise ValueError(f"unknown action_mode {args.action_mode!r}")


def eval_robomme(args: Args) -> None:
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    if args.dataset not in EPISODE_LIMITS:
        raise ValueError(f"dataset must be one of {sorted(EPISODE_LIMITS)}")
    tasks = resolve_tasks(args.tasks)

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    decisions_path = out_dir / "decisions.jsonl"
    for record_path in (episodes_path, decisions_path):
        record_path.write_text("")

    env_memory_mode = os.environ.get("MEMORY_MODE")
    memory_params = memory_params_from_env()
    git_sha = read_git_sha(pathlib.Path(__file__).resolve().parent)
    ckpt_sha = checkpoint_sha(args.pretrained_path)

    model = RoboMMEPolicyClient(
        ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        image_size=args.resize_size,
    )
    stride = args.conditioning_stride or model.action_chunk_size

    total_episodes, total_successes = 0, 0
    for task_id in tasks:
        suite = f"robomme_{SUITE_OF[task_id].lower()}"
        builder = BenchmarkEnvBuilder(
            env_id=task_id,
            dataset=args.dataset,
            action_space="ee_pose",
            gui_render=False,
            max_steps=args.max_steps,
        )
        split_size = builder.get_episode_num() or EPISODE_LIMITS[args.dataset]
        last = min(args.episode_start + args.num_episodes, split_size)
        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.episode_start, last), desc=task_id):
            env = builder.make_env_for_episode(episode_idx)
            try:
                obs, info = env.reset()
                task_goal = str(info["task_goal"][0])
                model.reset(
                    task_description=task_goal,
                    episode_id=f"{task_id}--{TASK_INDEX[task_id]}--ep{episode_idx}",
                    task_key=f"{task_id}--{TASK_INDEX[task_id]}",
                )
                episode_seed = model.episode_seed

                decision_rows = []
                first_memory_extras = None

                # -------- conditioning video ingestion (memory writes) --------
                n_frames = len(obs["front_rgb_list"])
                conditioning_ingested = 0
                if args.ingest_conditioning and n_frames > 1:
                    demo_ids = list(range(0, n_frames - 1, stride))
                    for k, i in enumerate(demo_ids):
                        state_i = _build_state(obs, i) if args.with_state == "true" else None
                        model.ingest(
                            [_frame(obs["front_rgb_list"][i]), _frame(obs["wrist_rgb_list"][i])],
                            state_i,
                        )
                        conditioning_ingested += 1
                        if first_memory_extras is None:
                            first_memory_extras = model.last_memory_extras
                        decision_rows.append(decision_record(
                            episode_idx=episode_idx,
                            d=k - len(demo_ids),  # negative = pre-execution
                            memory_extras=model.last_memory_extras,
                            extras={"task_id": TASK_INDEX[task_id], "phase": "conditioning"},
                        ))

                # -------- execution loop --------
                replay_frames = []
                step = 0
                status = "ongoing"
                start_time = time.time()
                while step < args.max_steps:
                    front = _frame(obs["front_rgb_list"][-1])
                    wrist = _frame(obs["wrist_rgb_list"][-1])
                    eef_state = _vec(obs["eef_state_list"][-1], 6)
                    state = _build_state(obs) if args.with_state == "true" else None
                    replay_frames.append(np.hstack([front, wrist]))

                    raw = model.step([front, wrist], state, step)
                    if step % model.action_chunk_size == 0:
                        if first_memory_extras is None:
                            first_memory_extras = model.last_memory_extras
                        decision_rows.append(decision_record(
                            episode_idx=episode_idx,
                            d=step // model.action_chunk_size,
                            memory_extras=model.last_memory_extras,
                            extras={"task_id": TASK_INDEX[task_id], "phase": "execution"},
                        ))

                    action = to_robomme_action(
                        raw, eef_state, args, no_gripper=task_id in NO_GRIPPER_TASKS)
                    obs, _, terminated, truncated, info = env.step(action)
                    status = str(info.get("status", "unknown"))
                    step += 1
                    if status == "error":
                        logging.warning(f"{task_id} ep{episode_idx} env error: "
                                        f"{info.get('error_message', 'unknown')}")
                        break
                    if bool(_to_numpy(terminated).any()) or bool(_to_numpy(truncated).any()):
                        break

                success = status == "success"
                task_episodes += 1
                total_episodes += 1
                task_successes += int(success)
                total_successes += int(success)
                num_decisions = sum(1 for r in decision_rows if r.get("phase") == "execution")

                suffix = "success" if success else f"failure-{status}"
                imageio.mimwrite(
                    out_dir / f"rollout_{task_id}_ep{episode_idx}_{suffix}.mp4",
                    replay_frames, fps=30,
                )
                logging.info(
                    f"{task_id} ep{episode_idx}: status={status} steps={step} "
                    f"decisions={num_decisions} cond={conditioning_ingested} "
                    f"wall={time.time() - start_time:.1f}s | "
                    f"running {total_successes}/{total_episodes}"
                )

                server_extras = first_memory_extras or {}
                record = episode_record(
                    suite=suite,
                    task_id=TASK_INDEX[task_id],
                    task_description=task_goal,
                    episode_idx=episode_idx,
                    memory_mode=resolve_memory_mode(env_memory_mode, server_extras.get("mode")),
                    memory_params=memory_params,
                    episode_seed=episode_seed,
                    success=success,
                    num_env_steps=step,
                    num_decisions=num_decisions,
                    ckpt=args.pretrained_path,
                    ckpt_sha=ckpt_sha,
                    git_sha=git_sha,
                    extras={
                        "task_name": task_id,
                        "benchmark": "robomme",
                        "dataset_split": args.dataset,
                        "status": status,
                        "action_mode": args.action_mode,
                        "conditioning_frames": n_frames - 1,
                        "conditioning_ingested": conditioning_ingested,
                        "donor_source": server_extras.get("donor_episode"),
                    },
                )
                with open(episodes_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if decision_rows:
                    with open(decisions_path, "a") as f:
                        f.writelines(json.dumps(row) + "\n" for row in decision_rows)
            finally:
                env.close()

        logging.info(
            f"[{task_id}] success rate: {task_successes}/{task_episodes} "
            f"({(task_successes / task_episodes * 100) if task_episodes else 0:.1f}%)"
        )

    logging.info(f"Total success rate: {total_successes}/{total_episodes} "
                 f"({(total_successes / total_episodes * 100) if total_episodes else 0:.1f}%)")
    logging.info(f"Total episodes: {total_episodes}")


if __name__ == "__main__":
    tyro.cli(eval_robomme)
