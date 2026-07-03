# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

import numpy as np
import torch

import websockets.asyncio.server
import websockets.frames

from starVLA.model.modules.memory.state import MemoryState

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools


_MEMORY_MODES = {
    "live",
    "prior",
    "bypass",
    "reset_k",
    "freeze_k",
    "write_every",
    "foreign",
    "noisematch",
    "permute_once",
    "noreset",
}
_DUMP_FILE_PATTERN = re.compile(r"d(\d+)\.pt")
# Structured episode ids are "<suite>--<task_id>--ep<episode_idx>"; the prefix
# before the trailing "--ep<idx>" is the task key used for donor exclusion.
_EPISODE_ID_PATTERN = re.compile(r"(?P<task_key>.+)--ep\d+$")


def _task_key_from_episode_id(episode_id: str) -> Optional[str]:
    match = _EPISODE_ID_PATTERN.match(episode_id)
    return match.group("task_key") if match else None


@dataclass(frozen=True)
class MemoryServerConfig:
    """Serve-time memory-mode policy, parsed once from the environment."""

    mode: str = "live"
    reset_k: Optional[int] = None
    freeze_k: Optional[int] = None
    write_every: Optional[int] = None
    gate_scale: Optional[float] = None
    donor_dir: Optional[str] = None
    state_dump_dir: Optional[str] = None
    counterfactual: bool = False
    permute_at: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode == "zero":
            logging.warning("MEMORY_MODE=zero is deprecated; use 'prior'")
            object.__setattr__(self, "mode", "prior")
        if self.mode not in _MEMORY_MODES:
            raise ValueError(f"unsupported memory_mode: {self.mode}")
        if self.mode == "reset_k" and (self.reset_k is None or self.reset_k < 1):
            raise ValueError("reset_k mode requires MEMORY_RESET_K >= 1")
        if self.mode == "freeze_k" and (self.freeze_k is None or self.freeze_k < 0):
            raise ValueError("freeze_k mode requires MEMORY_FREEZE_K >= 0")
        if self.mode == "write_every" and (self.write_every is None or self.write_every < 1):
            raise ValueError("write_every mode requires MEMORY_WRITE_EVERY >= 1")
        if self.mode in {"foreign", "noisematch"} and not self.donor_dir:
            raise ValueError(f"{self.mode} mode requires MEMORY_DONOR_DIR")
        if self.mode == "permute_once":
            if self.permute_at is None:
                object.__setattr__(self, "permute_at", 4)
            elif self.permute_at < 1:
                raise ValueError("permute_once mode requires MEMORY_PERMUTE_AT >= 1")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "MemoryServerConfig":
        env = os.environ if env is None else env

        def _int(name: str) -> Optional[int]:
            value = env.get(name)
            return int(value) if value else None

        counterfactual = env.get("MEMORY_COUNTERFACTUAL", "0")
        if counterfactual not in {"0", "1"}:
            raise ValueError("MEMORY_COUNTERFACTUAL must be '0' or '1'")
        gate_scale = env.get("MEMORY_GATE_SCALE")
        return cls(
            mode=env.get("MEMORY_MODE", "live"),
            reset_k=_int("MEMORY_RESET_K"),
            freeze_k=_int("MEMORY_FREEZE_K"),
            write_every=_int("MEMORY_WRITE_EVERY"),
            gate_scale=float(gate_scale) if gate_scale else None,
            donor_dir=env.get("MEMORY_DONOR_DIR") or None,
            state_dump_dir=env.get("MEMORY_STATE_DUMP_DIR") or None,
            counterfactual=counterfactual == "1",
            permute_at=_int("MEMORY_PERMUTE_AT"),
        )


@dataclass(frozen=True)
class _DecisionPlan:
    """predict_action kwarg schedule for one decision; pure in (config, counter)."""

    memory_state_override: Optional[str]  # None=session state | prior | donor | noise | permute
    update_memory: bool
    memory_bypass: bool
    clear_state_on_reset: bool


def plan_memory_decision(config: MemoryServerConfig, counter: int) -> _DecisionPlan:
    """Map (mode, params, per-session decision counter) to one decision's kwargs."""

    mode = config.mode
    if mode == "prior":
        return _DecisionPlan("prior", False, False, True)
    if mode == "bypass":
        return _DecisionPlan(None, False, True, True)
    if mode == "reset_k":
        override = "prior" if counter % config.reset_k == 0 else None
        return _DecisionPlan(override, True, False, True)
    if mode == "freeze_k":
        return _DecisionPlan(None, counter < config.freeze_k, False, True)
    if mode == "write_every":
        return _DecisionPlan(None, counter % config.write_every == 0, False, True)
    if mode == "foreign":
        return _DecisionPlan("donor", False, False, True)
    if mode == "noisematch":
        return _DecisionPlan("noise", False, False, True)
    if mode == "permute_once":
        override = "permute" if counter == config.permute_at else None
        return _DecisionPlan(override, True, False, True)
    if mode == "noreset":
        return _DecisionPlan(None, True, False, False)
    return _DecisionPlan(None, True, False, True)  # live


def _scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean())
    return float(value)


def _state_payload(state: MemoryState, decision_index: int) -> dict:
    return {
        "working": state.working.detach().cpu(),
        "episodic": state.episodic.detach().cpu() if state.episodic is not None else None,
        "steps": state.steps.detach().cpu(),
        "valid": state.valid.detach().cpu(),
        "decision_index": decision_index,
    }


def _load_memory_state(path: Path, device: torch.device) -> MemoryState:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, MemoryState):
        return payload.detach().to(device=device)
    episodic = payload.get("episodic")
    return MemoryState(
        working=payload["working"].to(device=device, dtype=torch.float32),
        episodic=episodic.to(device=device, dtype=torch.float32) if episodic is not None else None,
        steps=payload["steps"].to(device=device),
        valid=payload["valid"].to(device=device),
    )


def _permute_slots(state: MemoryState) -> MemoryState:
    """Cyclically roll the working slots (a derangement: no slot keeps its index).

    Fusion reads are permutation-invariant over slots (slot_ids enter only the
    write query), so a permutation is only meaningful when its written successor
    is committed back into the state chain: permute_once applies this exactly
    once, at MEMORY_PERMUTE_AT, to probe write-side slot-binding consistency.
    """

    return MemoryState(
        working=state.working.roll(shifts=1, dims=1),
        episodic=state.episodic,
        steps=state.steps,
        valid=state.valid,
    )


class _DonorBank:
    """Read-only donor states laid out as MEMORY_STATE_DUMP_DIR writes them.

    Accepts a single state file, one episode directory of ``d<idx>.pt`` files,
    or a directory of such episode directories. Episode directories named
    ``<suite>--<task_id>--ep<idx>`` carry the task key used to exclude
    same-task donors in foreign mode.
    """

    def __init__(self, root: str) -> None:
        self._episodes = self._index(Path(root))
        if not self._episodes:
            raise ValueError(f"no donor states found under {root}")
        self._task_keys = [_task_key_from_episode_id(name) for name, _ in self._episodes]

    @staticmethod
    def _index(root: Path) -> List[Tuple[str, List[Path]]]:
        if root.is_file():
            return [(root.stem, [root])]
        if not root.is_dir():
            raise ValueError(f"donor path does not exist: {root}")

        def _states(directory: Path) -> List[Path]:
            matches = []
            for child in directory.iterdir():
                match = _DUMP_FILE_PATTERN.fullmatch(child.name)
                if match and child.is_file():
                    matches.append((int(match.group(1)), child))
            return [path for _, path in sorted(matches)]

        direct = _states(root)
        if direct:
            return [(root.name, direct)]
        episodes = [(child.name, _states(child)) for child in sorted(root.iterdir()) if child.is_dir()]
        return [(name, files) for name, files in episodes if files]

    @property
    def has_task_metadata(self) -> bool:
        return any(key is not None for key in self._task_keys)

    def episode_name(self, episode: int) -> str:
        return self._episodes[episode][0]

    def pick_episode(self, seed: int, task_key: Optional[str] = None) -> int:
        """Deterministic donor pick, excluding donors from the recipient's task."""

        eligible = [
            index
            for index, donor_key in enumerate(self._task_keys)
            if task_key is None or donor_key is None or donor_key != task_key
        ]
        if not eligible:
            raise ValueError(f"donor bank has no cross-task donor for task {task_key!r}")
        return eligible[seed % len(eligible)]

    def write_index(self, episode: int, decision: int) -> Optional[int]:
        """Donor file injected at recipient decision ``d`` under the maturity
        convention: live at decision d reads a state with d absorbed writes
        (decisions 0..d-1) and dump files are post-write (``d<i>.pt`` = state
        after decision i's write), so decision d injects file d<d-1>, clamped
        to the donor's last file; decision 0 reads the prior (None)."""

        if decision == 0:
            return None
        files = self._episodes[episode][1]
        return min(decision - 1, len(files) - 1)

    def state_for(self, episode: int, write_index: int, device: torch.device) -> MemoryState:
        return _load_memory_state(self._episodes[episode][1][write_index], device)

    def fit_moments(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-slot-channel mean/std of working states over the whole bank."""

        bank = torch.stack(
            [
                _load_memory_state(path, torch.device("cpu")).working[0]
                for _, episode in self._episodes
                for path in episode
            ]
        )
        return bank.mean(dim=0), bank.std(dim=0, unbiased=False)


@dataclass
class _ConnectionSession:
    """Private activation and RNG state owned by one WebSocket handler."""

    memory_state: object = None
    generator: torch.Generator | None = None
    episode_id: str | None = None
    batch_size: int | None = None
    ready: bool = False
    decision_index: int = 0
    donor_episode: int | None = None
    donor_state: object = None
    legacy_reset_warned: bool = False

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        memory_mode: str = "live",
        memory_config: MemoryServerConfig | None = None,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._memory_config = memory_config or MemoryServerConfig(mode=memory_mode)
        if not getattr(policy, "memory_enabled", False) and (
            self._memory_config.mode != "live"
            or self._memory_config.donor_dir
            or self._memory_config.state_dump_dir
            or self._memory_config.counterfactual
        ):
            raise RuntimeError(
                f"memory serve options configured (mode={self._memory_config.mode!r}, "
                f"donor_dir={self._memory_config.donor_dir!r}, "
                f"state_dump_dir={self._memory_config.state_dump_dir!r}, "
                f"counterfactual={self._memory_config.counterfactual}) but the served "
                "policy has no memory module; a wrong checkpoint must not fabricate "
                "a null result"
            )
        self._donor_bank = (
            _DonorBank(self._memory_config.donor_dir)
            if self._memory_config.mode in {"foreign", "noisematch"}
            else None
        )
        if self._memory_config.mode == "foreign" and not self._donor_bank.has_task_metadata:
            logging.warning(
                "MEMORY_DONOR_DIR=%s episode dirs carry no '<suite>--<task_id>--ep<idx>' "
                "task metadata; same-task donor exclusion is DISABLED for foreign mode",
                self._memory_config.donor_dir,
            )
        self._donor_moments = (
            self._donor_bank.fit_moments() if self._memory_config.mode == "noisematch" else None
        )
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=256 * 1024 * 1024,
            max_queue=4,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        session = _ConnectionSession()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg, session=session)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logging.exception("Unexpected WebSocket handler failure")
                await websocket.send("Internal server error")
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _policy_device(self) -> torch.device:
        try:
            return next(self._policy.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _sample_noise_state(self, episode_seed: int, device: torch.device) -> MemoryState:
        """Gaussian working state with per-slot-channel moments from the donor bank.

        By design this is ONE static per-episode draw, re-injected unchanged at
        every decision: noisematch is the statistics control (on-manifold
        moments, wrong content), not a maturity-matched stream like foreign.
        """

        mean, std = self._donor_moments
        noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(episode_seed))
        return MemoryState(
            working=(mean + std * noise).unsqueeze(0).to(device=device, dtype=torch.float32),
            episodic=None,
            steps=torch.zeros(1, dtype=torch.int64, device=device),
            valid=torch.ones(1, dtype=torch.bool, device=device),
        )

    def _resolve_memory_state(self, plan: _DecisionPlan, session: _ConnectionSession):
        if plan.memory_state_override is None:
            return session.memory_state
        if plan.memory_state_override == "prior":
            return None
        if plan.memory_state_override == "donor":
            write_index = self._donor_bank.write_index(
                session.donor_episode, session.decision_index
            )
            if write_index is None:
                return None  # decision 0 reads the learned prior, like live
            return self._donor_bank.state_for(
                session.donor_episode, write_index, self._policy_device()
            )
        if plan.memory_state_override == "noise":
            return session.donor_state
        # permute (one-shot): the rolled state's written successor is committed,
        # so the derangement enters the live chain exactly once.
        state = session.memory_state
        return _permute_slots(state) if state is not None else None

    def _sample_initial_noise(self, batch_size: int, session: _ConnectionSession) -> torch.Tensor:
        head_config = self._policy.action_model.config
        # Match the action head's internal draw dtype (policy parameter dtype;
        # bf16 under --use_bf16) so counterfactual-on runs stay draw-for-draw
        # paired with counterfactual-off runs.
        param = next(self._policy.parameters())
        return torch.randn(
            (batch_size, head_config.action_horizon, head_config.action_dim),
            device=param.device,
            dtype=param.dtype,
            generator=session.generator,
        )

    def _clear_module_diagnostics(self) -> None:
        # Null the per-forward side channels so skipped stages cannot report
        # a stale previous decision.
        fusion = getattr(self._policy, "policy_memory_fusion", None)
        if fusion is not None:
            fusion.last_fusion_diagnostics = None
        memory = getattr(self._policy, "memory_module", None)
        if memory is not None:
            memory.last_write_diagnostics = None

    def _memory_extras(self, session: _ConnectionSession) -> dict:
        read_diag = getattr(self._policy, "last_memory_diagnostics", None) or {}
        fusion = getattr(self._policy, "policy_memory_fusion", None)
        fusion_diag = getattr(fusion, "last_fusion_diagnostics", None) or {}
        memory = getattr(self._policy, "memory_module", None)
        write_diag = getattr(memory, "last_write_diagnostics", None) or {}
        extras = {
            "mode": self._memory_config.mode,
            "decision_index": session.decision_index,
            "working_norm": _scalar(read_diag.get("working_norm")),
            "injection_ratio": _scalar(fusion_diag.get("injection_ratio")),
            "update_gate_mean": _scalar(write_diag.get("update_gate_mean")),
        }
        if self._memory_config.mode == "permute_once":
            extras["permute_applied"] = session.decision_index == self._memory_config.permute_at
        if self._memory_config.mode == "foreign":
            donor = session.donor_episode
            extras["donor_episode"] = (
                None if donor is None else self._donor_bank.episode_name(donor)
            )
            extras["donor_decision"] = (
                None
                if donor is None
                else self._donor_bank.write_index(donor, session.decision_index)
            )
        return extras

    def _counterfactual_delta(self, call_payload: dict, live_output: dict) -> float:
        """Re-run the decision with memory bypassed, reusing tokens and noise."""

        cf_payload = dict(call_payload)
        cf_payload.update(
            memory_bypass=True,
            update_memory=False,
            return_memory_state=False,
            qwen_cache=self._policy.last_qwen_cache,
            keep_qwen_cache=False,
            generator=None,
        )
        cf_output = self._policy.predict_action(**cf_payload)
        delta = np.asarray(live_output["normalized_actions"], dtype=np.float64) - np.asarray(
            cf_output["normalized_actions"], dtype=np.float64
        )
        return float(np.linalg.norm(delta))

    def _dump_state(self, state: MemoryState, session: _ConnectionSession) -> None:
        # Donor convention: d<i>.pt = state after decision i's committed write;
        # the caller only dumps decisions whose write was committed. The episode
        # dir is the sanitized full episode_id ("<suite>--<task_id>--ep<idx>"),
        # so parallel suite servers sharing one dump dir cannot collide.
        episode = re.sub(r"[^A-Za-z0-9._-]+", "_", session.episode_id or "default")
        out_dir = Path(self._memory_config.state_dump_dir) / episode
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            _state_payload(state, session.decision_index),
            out_dir / f"d{session.decision_index}.pt",
        )

    def _route_message(self, msg: dict, session: _ConnectionSession | None = None) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": "default",
                "error": {"message": "Request must be a dict"},
            }
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload
        legacy_reset = mtype == "infer" and msg.get("reset") is True
        if legacy_reset:
            # Compatibility with the original flat reset request.
            mtype = "reset"
            payload = {
                "episode_id": msg.get("instruction", "legacy"),
                "episode_seed": msg.get("episode_seed", 0),
            }
        if session is None:
            session = _ConnectionSession()
        if legacy_reset and self._memory_config.mode != "live" and not session.legacy_reset_warned:
            session.legacy_reset_warned = True
            logging.warning(
                "legacy flat reset under memory mode %r: episode_seed/episode_id "
                "defaulted (seed=%s), so the per-episode noise/donor design is not honored",
                self._memory_config.mode,
                payload["episode_seed"],
            )

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype == "reset":
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_result",
                    "request_id": req_id,
                    "error": {"message": "Reset payload must be a dict"},
                }
            try:
                episode_seed = int(payload.get("episode_seed", 0))
                episode_id = str(payload.get("episode_id", "default"))
                device = self._policy_device()
                candidate_generator = torch.Generator(device=device)
                candidate_generator.manual_seed(episode_seed)
                donor_episode = None
                donor_state = None
                if self._memory_config.mode == "foreign":
                    task_key = payload.get("task_key") or _task_key_from_episode_id(episode_id)
                    donor_episode = self._donor_bank.pick_episode(episode_seed, task_key)
                elif self._memory_config.mode == "noisematch":
                    donor_state = self._sample_noise_state(episode_seed, device)
                # Commit reset state only after every validation/initialization
                # operation above has succeeded.
                session.generator = candidate_generator
                if plan_memory_decision(self._memory_config, 0).clear_state_on_reset:
                    session.memory_state = None
                session.episode_id = episode_id
                session.batch_size = None
                session.ready = True
                session.decision_index = 0
                session.donor_episode = donor_episode
                session.donor_state = donor_state
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "reset_result",
                    "request_id": req_id,
                    "data": {"episode_id": episode_id, "episode_seed": episode_seed},
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_result",
                    "request_id": req_id,
                    "error": {"message": str(exc)},
                }

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                memory_enabled = bool(getattr(self._policy, "memory_enabled", False))
                if memory_enabled and not session.ready:
                    raise RuntimeError("memory-enabled inference requires an explicit reset")
                batch_images = payload.get("batch_images")
                if memory_enabled and len(batch_images) != 1:
                    raise ValueError("Phase-1 serving supports exactly one episode row per connection")
                if memory_enabled:
                    if session.batch_size is None:
                        session.batch_size = len(batch_images)
                    elif session.batch_size != len(batch_images):
                        raise ValueError("batch size changed inside a live session; reset first")

                call_payload = dict(payload)
                call_payload["batch_images"] = image_tools.to_pil_preserve(batch_images)
                suppress_write = bool(call_payload.pop("suppress_write", False))
                rng_before = (
                    session.generator.get_state().clone()
                    if session.generator is not None
                    else None
                )
                state_before = session.memory_state
                if memory_enabled:
                    plan = plan_memory_decision(self._memory_config, session.decision_index)
                    update_memory = plan.update_memory and not suppress_write
                    call_payload.update(
                        memory_state=self._resolve_memory_state(plan, session),
                        return_memory_state=True,
                        update_memory=update_memory,
                        memory_bypass=plan.memory_bypass,
                        generator=session.generator,
                    )
                    if self._memory_config.counterfactual:
                        call_payload.update(
                            initial_noise=self._sample_initial_noise(len(batch_images), session),
                            keep_qwen_cache=True,
                        )
                    self._clear_module_diagnostics()
                    output_dict, candidate_state = self._policy.predict_action(**call_payload)
                    memory_extras = self._memory_extras(session)
                    if self._memory_config.counterfactual:
                        memory_extras["cf_delta_action_l2"] = self._counterfactual_delta(
                            call_payload, output_dict
                        )
                    if update_memory:
                        session.memory_state = candidate_state
                        # Dump only committed writes: skipped/suppressed decisions
                        # must not pollute the donor bank with stale or injected
                        # states masquerading as post-write states.
                        if self._memory_config.state_dump_dir and candidate_state is not None:
                            self._dump_state(candidate_state, session)
                    output_dict["memory_extras"] = memory_extras
                    session.decision_index += 1
                else:
                    if session.generator is not None:
                        call_payload["generator"] = session.generator
                    output_dict = self._policy.predict_action(**call_payload)
            except Exception as e:
                if session.generator is not None and "rng_before" in locals() and rng_before is not None:
                    session.generator.set_state(rng_before)
                if "state_before" in locals():
                    session.memory_state = state_before
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
