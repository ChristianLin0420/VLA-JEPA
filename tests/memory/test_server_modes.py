import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from deployment.model_server.tools.websocket_policy_server import (
    MemoryServerConfig,
    WebsocketPolicyServer,
    _ConnectionSession,
    _DonorBank,
    _permute_slots,
    plan_memory_decision,
)
from starVLA.model.modules.memory.state import MemoryState

EPISODE_DECISIONS = 12


def _make_state(fill: float, slots: int = 4, dim: int = 3) -> MemoryState:
    return MemoryState(
        working=torch.full((1, slots, dim), float(fill), dtype=torch.float32),
        episodic=None,
        steps=torch.zeros(1, dtype=torch.int64),
        valid=torch.ones(1, dtype=torch.bool),
    )


def _schedule(config: MemoryServerConfig):
    return [plan_memory_decision(config, d) for d in range(EPISODE_DECISIONS)]


class _RecordingPolicy(nn.Module):
    """Fake memory policy recording per-call kwargs; state value counts writes."""

    memory_enabled = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []
        self.last_qwen_cache = None
        self.action_model = SimpleNamespace(
            config=SimpleNamespace(action_horizon=2, action_dim=3)
        )

    def predict_action(
        self,
        *,
        memory_state=None,
        update_memory=True,
        memory_bypass=False,
        generator=None,
        initial_noise=None,
        qwen_cache=None,
        keep_qwen_cache=False,
        return_memory_state=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "memory_state": memory_state,
                "update_memory": update_memory,
                "memory_bypass": memory_bypass,
                "qwen_cache": qwen_cache,
                "keep_qwen_cache": keep_qwen_cache,
                "initial_noise": initial_noise,
                "kwargs": kwargs,
            }
        )
        self.last_qwen_cache = ("qwen", len(self.calls)) if keep_qwen_cache else None
        if memory_bypass:
            return (
                ({"normalized_actions": np.zeros((1, 2, 3), dtype=np.float32)}, memory_state)
                if return_memory_state
                else {"normalized_actions": np.zeros((1, 2, 3), dtype=np.float32)}
            )
        previous = 0.0 if memory_state is None else float(memory_state.working[0, 0, 0])
        candidate = _make_state(previous + 1.0) if update_memory else memory_state
        output = {"normalized_actions": np.full((1, 2, 3), previous + 1.0, dtype=np.float32)}
        return (output, candidate) if return_memory_state else output


class _StatelessPolicy(nn.Module):
    memory_enabled = False

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def predict_action(self, *, batch_images, **kwargs):
        return {"normalized_actions": np.zeros((len(batch_images), 1, 1), dtype=np.float32)}


def _reset(server, session, seed=7, episode_id="episode", task_key=None):
    payload = {"episode_id": episode_id, "episode_seed": seed}
    if task_key is not None:
        payload["task_key"] = task_key
    return server._route_message(
        {"type": "reset", "request_id": "reset", "payload": payload},
        session=session,
    )


def _save_bank_state(directory: Path, index: int, fill: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    state = _make_state(fill)
    torch.save(
        {
            "working": state.working,
            "episodic": None,
            "steps": state.steps,
            "valid": state.valid,
            "decision_index": index,
        },
        directory / f"d{index}.pt",
    )


def _infer(server, session, **extra):
    payload = {
        "batch_images": [[Image.new("RGB", (2, 2))]],
        "instructions": ["test"],
    }
    payload.update(extra)
    return server._route_message(
        {"type": "infer", "request_id": "infer", "payload": payload},
        session=session,
    )


class DecisionPlanTest(unittest.TestCase):
    def test_live_schedule(self):
        for plan in _schedule(MemoryServerConfig(mode="live")):
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                (None, True, False),
            )
            self.assertTrue(plan.clear_state_on_reset)

    def test_prior_schedule(self):
        for plan in _schedule(MemoryServerConfig(mode="prior")):
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                ("prior", False, False),
            )

    def test_bypass_schedule(self):
        for plan in _schedule(MemoryServerConfig(mode="bypass")):
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                (None, False, True),
            )

    def test_reset_k_schedule(self):
        plans = _schedule(MemoryServerConfig(mode="reset_k", reset_k=4))
        self.assertEqual(
            [plan.memory_state_override for plan in plans],
            ["prior", None, None, None] * 3,
        )
        self.assertTrue(all(plan.update_memory and not plan.memory_bypass for plan in plans))

    def test_reset_k_of_one_matches_prior_state_schedule(self):
        plans = _schedule(MemoryServerConfig(mode="reset_k", reset_k=1))
        self.assertTrue(all(plan.memory_state_override == "prior" for plan in plans))

    def test_freeze_k_schedule(self):
        plans = _schedule(MemoryServerConfig(mode="freeze_k", freeze_k=3))
        self.assertEqual([plan.update_memory for plan in plans], [True] * 3 + [False] * 9)

    def test_freeze_k_of_zero_never_writes(self):
        plans = _schedule(MemoryServerConfig(mode="freeze_k", freeze_k=0))
        self.assertFalse(any(plan.update_memory for plan in plans))

    def test_write_every_schedule(self):
        plans = _schedule(MemoryServerConfig(mode="write_every", write_every=4))
        self.assertEqual(
            [plan.update_memory for plan in plans],
            [d % 4 == 0 for d in range(EPISODE_DECISIONS)],
        )

    def test_write_every_of_one_writes_every_decision(self):
        plans = _schedule(MemoryServerConfig(mode="write_every", write_every=1))
        self.assertTrue(all(plan.update_memory for plan in plans))

    def test_foreign_and_noisematch_schedules(self):
        for mode, override in (("foreign", "donor"), ("noisematch", "noise")):
            for plan in _schedule(MemoryServerConfig(mode=mode, donor_dir="donor")):
                self.assertEqual(
                    (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                    (override, False, False),
                )

    def test_permute_once_schedule(self):
        plans = _schedule(MemoryServerConfig(mode="permute_once", permute_at=3))
        for counter, plan in enumerate(plans):
            expected = "permute" if counter == 3 else None
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                (expected, True, False),
            )

    def test_permute_once_defaults_permute_at(self):
        self.assertEqual(MemoryServerConfig(mode="permute_once").permute_at, 4)
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="permute_once", permute_at=0)

    def test_noreset_schedule_keeps_state_across_resets(self):
        for plan in _schedule(MemoryServerConfig(mode="noreset")):
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                (None, True, False),
            )
            self.assertFalse(plan.clear_state_on_reset)


class MemoryServerConfigTest(unittest.TestCase):
    def test_zero_alias_warns_and_maps_to_prior(self):
        with self.assertLogs(level="WARNING") as logs:
            config = MemoryServerConfig(mode="zero")
        self.assertEqual(config.mode, "prior")
        self.assertTrue(any("deprecated" in line for line in logs.output))

    def test_invalid_mode_and_missing_params_raise(self):
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="banana")
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="reset_k")
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="reset_k", reset_k=0)
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="freeze_k", freeze_k=-1)
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="write_every", write_every=0)
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="foreign")
        with self.assertRaises(ValueError):
            MemoryServerConfig(mode="noisematch")

    def test_from_env_parses_all_params(self):
        config = MemoryServerConfig.from_env(
            {
                "MEMORY_MODE": "reset_k",
                "MEMORY_RESET_K": "8",
                "MEMORY_FREEZE_K": "2",
                "MEMORY_WRITE_EVERY": "4",
                "MEMORY_GATE_SCALE": "2.0",
                "MEMORY_DONOR_DIR": "/tmp/donors",
                "MEMORY_STATE_DUMP_DIR": "/tmp/dumps",
                "MEMORY_COUNTERFACTUAL": "1",
            }
        )
        self.assertEqual(config.mode, "reset_k")
        self.assertEqual(config.reset_k, 8)
        self.assertEqual(config.freeze_k, 2)
        self.assertEqual(config.write_every, 4)
        self.assertEqual(config.gate_scale, 2.0)
        self.assertEqual(config.donor_dir, "/tmp/donors")
        self.assertEqual(config.state_dump_dir, "/tmp/dumps")
        self.assertTrue(config.counterfactual)

    def test_from_env_defaults_and_bad_counterfactual(self):
        config = MemoryServerConfig.from_env({})
        self.assertEqual(config.mode, "live")
        self.assertFalse(config.counterfactual)
        with self.assertRaises(ValueError):
            MemoryServerConfig.from_env({"MEMORY_COUNTERFACTUAL": "2"})


class ServerDispatchTest(unittest.TestCase):
    def test_suppress_write_forces_single_decision_freeze(self):
        policy = _RecordingPolicy()
        server = WebsocketPolicyServer(policy)
        session = _ConnectionSession()
        _reset(server, session)

        self.assertTrue(_infer(server, session, suppress_write=True)["ok"])
        self.assertFalse(policy.calls[0]["update_memory"])
        self.assertNotIn("suppress_write", policy.calls[0]["kwargs"])
        self.assertIsNone(session.memory_state)

        self.assertTrue(_infer(server, session)["ok"])
        self.assertTrue(policy.calls[1]["update_memory"])
        self.assertEqual(float(session.memory_state.working[0, 0, 0]), 1.0)
        self.assertEqual(session.decision_index, 2)

    def test_reset_k_server_injects_prior_state_periodically(self):
        policy = _RecordingPolicy()
        config = MemoryServerConfig(mode="reset_k", reset_k=2)
        server = WebsocketPolicyServer(policy, memory_config=config)
        session = _ConnectionSession()
        _reset(server, session)
        for _ in range(4):
            self.assertTrue(_infer(server, session)["ok"])
        passed = [call["memory_state"] is None for call in policy.calls]
        self.assertEqual(passed, [True, False, True, False])

    def test_freeze_k_server_schedule_and_commit(self):
        policy = _RecordingPolicy()
        config = MemoryServerConfig(mode="freeze_k", freeze_k=2)
        server = WebsocketPolicyServer(policy, memory_config=config)
        session = _ConnectionSession()
        _reset(server, session)
        for _ in range(4):
            self.assertTrue(_infer(server, session)["ok"])
        self.assertEqual(
            [call["update_memory"] for call in policy.calls], [True, True, False, False]
        )
        self.assertEqual(float(session.memory_state.working[0, 0, 0]), 2.0)

    def test_memory_extras_present_only_for_memory_policies(self):
        policy = _RecordingPolicy()
        server = WebsocketPolicyServer(policy)
        session = _ConnectionSession()
        _reset(server, session)
        first = _infer(server, session)["data"]["memory_extras"]
        second = _infer(server, session)["data"]["memory_extras"]
        self.assertEqual(first["mode"], "live")
        self.assertEqual((first["decision_index"], second["decision_index"]), (0, 1))
        self.assertIn("working_norm", first)
        self.assertIn("injection_ratio", first)
        self.assertIn("update_gate_mean", first)

        stateless = WebsocketPolicyServer(_StatelessPolicy())
        out = _infer(stateless, _ConnectionSession())
        self.assertNotIn("memory_extras", out["data"])

    def test_counterfactual_second_call_reuses_cache_and_noise(self):
        policy = _RecordingPolicy()
        config = MemoryServerConfig(mode="live", counterfactual=True)
        server = WebsocketPolicyServer(policy, memory_config=config)
        session = _ConnectionSession()
        _reset(server, session)

        out = _infer(server, session)
        self.assertTrue(out["ok"])
        self.assertEqual(len(policy.calls), 2)
        fused, bypassed = policy.calls
        self.assertTrue(fused["keep_qwen_cache"])
        self.assertFalse(fused["memory_bypass"])
        self.assertTrue(bypassed["memory_bypass"])
        self.assertFalse(bypassed["update_memory"])
        self.assertEqual(bypassed["qwen_cache"], ("qwen", 1))
        torch.testing.assert_close(fused["initial_noise"], bypassed["initial_noise"])
        extras = out["data"]["memory_extras"]
        self.assertIsInstance(extras["cf_delta_action_l2"], float)
        self.assertAlmostEqual(extras["cf_delta_action_l2"], float(np.sqrt(6.0)), places=5)
        # the LIVE action and candidate state are the ones committed
        self.assertEqual(float(session.memory_state.working[0, 0, 0]), 1.0)
        self.assertEqual(session.decision_index, 1)

    def test_state_dump_and_foreign_donor_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = _RecordingPolicy()
            config = MemoryServerConfig(mode="live", state_dump_dir=tmp)
            server = WebsocketPolicyServer(policy, memory_config=config)
            session = _ConnectionSession()
            _reset(server, session, episode_id="libero_10--3--ep0")
            for _ in range(3):
                self.assertTrue(_infer(server, session)["ok"])
            episode_dir = Path(tmp) / "libero_10--3--ep0"
            self.assertEqual(
                sorted(p.name for p in episode_dir.iterdir()), ["d0.pt", "d1.pt", "d2.pt"]
            )

            donor_policy = _RecordingPolicy()
            donor_config = MemoryServerConfig(mode="foreign", donor_dir=tmp)
            donor_server = WebsocketPolicyServer(donor_policy, memory_config=donor_config)
            donor_session = _ConnectionSession()
            _reset(donor_server, donor_session)
            responses = [_infer(donor_server, donor_session) for _ in range(5)]
            self.assertTrue(all(out["ok"] for out in responses))
            # Maturity convention: decision d injects donor file d<d-1> (the
            # post-write state of decision d-1, i.e. what a live read at d has
            # absorbed), clamped to the donor's last file; d=0 reads the prior.
            self.assertIsNone(donor_policy.calls[0]["memory_state"])
            injected = [
                float(call["memory_state"].working[0, 0, 0])
                for call in donor_policy.calls[1:]
            ]
            self.assertEqual(injected, [1.0, 2.0, 3.0, 3.0])
            self.assertFalse(any(call["update_memory"] for call in donor_policy.calls))
            self.assertIsNone(donor_session.memory_state)
            extras = [out["data"]["memory_extras"] for out in responses]
            self.assertEqual(
                [e["donor_episode"] for e in extras], ["libero_10--3--ep0"] * 5
            )
            self.assertEqual([e["donor_decision"] for e in extras], [None, 0, 1, 2, 2])

    def test_foreign_excludes_same_task_donors(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_bank_state(Path(tmp) / "libero_10--3--ep0", 0, 1.0)
            _save_bank_state(Path(tmp) / "libero_10--5--ep0", 0, 5.0)
            policy = _RecordingPolicy()
            server = WebsocketPolicyServer(
                policy, memory_config=MemoryServerConfig(mode="foreign", donor_dir=tmp)
            )
            session = _ConnectionSession()
            # seed 0 would pick the first (same-task) episode without exclusion;
            # task_key is derived from the structured episode_id.
            _reset(server, session, seed=0, episode_id="libero_10--3--ep9")
            _infer(server, session)
            out = _infer(server, session)
            self.assertEqual(float(policy.calls[1]["memory_state"].working[0, 0, 0]), 5.0)
            self.assertEqual(
                out["data"]["memory_extras"]["donor_episode"], "libero_10--5--ep0"
            )
            # An explicit task_key in the reset payload wins over parsing.
            explicit = _ConnectionSession()
            _reset(server, explicit, seed=0, episode_id="opaque", task_key="libero_10--5")
            _infer(server, explicit)
            _infer(server, explicit)
            self.assertEqual(
                float(policy.calls[-1]["memory_state"].working[0, 0, 0]), 1.0
            )
            # A bank whose only episodes share the recipient's task cannot serve.
            solo = _ConnectionSession()
            rejected = _reset(server, solo, seed=0, episode_id="x", task_key=None)
            self.assertTrue(rejected["ok"])  # no exclusion without a task key
            single_task = WebsocketPolicyServer(
                _RecordingPolicy(),
                memory_config=MemoryServerConfig(
                    mode="foreign", donor_dir=str(Path(tmp) / "libero_10--3--ep0")
                ),
            )
            failed = _reset(single_task, _ConnectionSession(), episode_id="libero_10--3--ep1")
            self.assertFalse(failed["ok"])
            self.assertIn("cross-task", failed["error"]["message"])

    def test_foreign_bank_without_task_metadata_warns_and_skips_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_bank_state(Path(tmp) / "libero-1", 0, 1.0)
            _save_bank_state(Path(tmp) / "libero-2", 0, 2.0)
            policy = _RecordingPolicy()
            with self.assertLogs(level="WARNING") as logs:
                server = WebsocketPolicyServer(
                    policy, memory_config=MemoryServerConfig(mode="foreign", donor_dir=tmp)
                )
            self.assertTrue(any("task metadata" in line for line in logs.output))
            session = _ConnectionSession()
            _reset(server, session, seed=0, episode_id="libero_10--3--ep0")
            _infer(server, session)
            _infer(server, session)
            # no exclusion possible: seed 0 picks the first episode
            self.assertEqual(float(policy.calls[1]["memory_state"].working[0, 0, 0]), 1.0)

    def test_state_dump_only_on_committed_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = _RecordingPolicy()
            config = MemoryServerConfig(mode="write_every", write_every=2, state_dump_dir=tmp)
            server = WebsocketPolicyServer(policy, memory_config=config)
            session = _ConnectionSession()
            _reset(server, session, episode_id="libero_10--0--ep0")
            for _ in range(4):
                self.assertTrue(_infer(server, session)["ok"])
            episode_dir = Path(tmp) / "libero_10--0--ep0"
            # d<i> = state after decision i's committed write; skipped decisions dump nothing
            self.assertEqual(sorted(p.name for p in episode_dir.iterdir()), ["d0.pt", "d2.pt"])

            live = WebsocketPolicyServer(
                _RecordingPolicy(),
                memory_config=MemoryServerConfig(mode="live", state_dump_dir=tmp),
            )
            suppressed = _ConnectionSession()
            _reset(live, suppressed, episode_id="libero_10--0--ep1")
            self.assertTrue(_infer(live, suppressed, suppress_write=True)["ok"])
            self.assertTrue(_infer(live, suppressed)["ok"])
            episode_dir = Path(tmp) / "libero_10--0--ep1"
            self.assertEqual(sorted(p.name for p in episode_dir.iterdir()), ["d1.pt"])

    def test_non_live_config_requires_memory_policy(self):
        for config in (
            MemoryServerConfig(mode="prior"),
            MemoryServerConfig(mode="live", counterfactual=True),
            MemoryServerConfig(mode="live", state_dump_dir="/tmp/dumps"),
            MemoryServerConfig(mode="live", donor_dir="/tmp/donors"),
        ):
            with self.assertRaises(RuntimeError):
                WebsocketPolicyServer(_StatelessPolicy(), memory_config=config)
        WebsocketPolicyServer(_StatelessPolicy())  # plain live serving stays legal

    def test_counterfactual_noise_matches_policy_param_dtype(self):
        policy = _RecordingPolicy().to(torch.bfloat16)
        config = MemoryServerConfig(mode="live", counterfactual=True)
        server = WebsocketPolicyServer(policy, memory_config=config)
        session = _ConnectionSession()
        _reset(server, session)
        self.assertTrue(_infer(server, session)["ok"])
        self.assertEqual(policy.calls[0]["initial_noise"].dtype, torch.bfloat16)

    def test_legacy_flat_reset_warns_once_per_session_for_non_live_modes(self):
        server = WebsocketPolicyServer(
            _RecordingPolicy(), memory_config=MemoryServerConfig(mode="prior")
        )
        session = _ConnectionSession()
        with self.assertLogs(level="WARNING") as logs:
            self.assertTrue(
                server._route_message({"reset": True, "instruction": "legacy"}, session=session)["ok"]
            )
        self.assertTrue(any("legacy flat reset" in line for line in logs.output))
        with self.assertNoLogs(level="WARNING"):
            server._route_message({"reset": True, "instruction": "legacy"}, session=session)
        live = WebsocketPolicyServer(_RecordingPolicy())
        with self.assertNoLogs(level="WARNING"):
            live._route_message(
                {"reset": True, "instruction": "legacy"}, session=_ConnectionSession()
            )

    def test_noisematch_state_is_seeded_and_moment_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank_dir = Path(tmp) / "ep0"
            bank_dir.mkdir()
            for idx, fill in enumerate((1.0, 3.0)):
                state = _make_state(fill)
                torch.save(
                    {
                        "working": state.working,
                        "episodic": None,
                        "steps": state.steps,
                        "valid": state.valid,
                        "decision_index": idx,
                    },
                    bank_dir / f"d{idx}.pt",
                )
            mean, std = _DonorBank(tmp).fit_moments()
            torch.testing.assert_close(mean, torch.full((4, 3), 2.0))
            torch.testing.assert_close(std, torch.full((4, 3), 1.0))

            policy = _RecordingPolicy()
            config = MemoryServerConfig(mode="noisematch", donor_dir=tmp)
            server = WebsocketPolicyServer(policy, memory_config=config)
            session = _ConnectionSession()
            _reset(server, session, seed=3)
            _infer(server, session)
            _infer(server, session)
            first, second = (call["memory_state"] for call in policy.calls)
            self.assertIs(first, second)  # persistent per-episode noise state
            self.assertEqual(first.working.shape, (1, 4, 3))

            repeat = _ConnectionSession()
            _reset(server, repeat, seed=3)
            _infer(server, repeat)
            torch.testing.assert_close(policy.calls[-1]["memory_state"].working, first.working)
            other = _ConnectionSession()
            _reset(server, other, seed=4)
            _infer(server, other)
            self.assertFalse(
                torch.equal(policy.calls[-1]["memory_state"].working, first.working)
            )

    def test_permute_once_is_a_derangement(self):
        state = _make_state(0.0, slots=8, dim=2)
        state.working.copy_(torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 2))
        permuted = _permute_slots(state)
        order = permuted.working[0, :, 0]
        # No slot keeps its index, content multiset preserved, deterministic.
        self.assertFalse(bool((order == state.working[0, :, 0]).any()))
        torch.testing.assert_close(order.sort().values, state.working[0, :, 0])
        torch.testing.assert_close(_permute_slots(state).working, permuted.working)

    def test_permute_once_enters_committed_chain_exactly_once(self):
        policy = _RecordingPolicy()
        server = WebsocketPolicyServer(
            policy, memory_config=MemoryServerConfig(mode="permute_once", permute_at=1)
        )
        session = _ConnectionSession()
        _reset(server, session)
        _infer(server, session)
        self.assertIsNone(policy.calls[0]["memory_state"])  # live before permute_at
        committed = session.memory_state
        _infer(server, session)  # decision 1: rolled state read AND written
        torch.testing.assert_close(
            policy.calls[1]["memory_state"].working, _permute_slots(committed).working
        )
        after_permute = session.memory_state
        _infer(server, session)  # decision 2: live again, no second roll
        torch.testing.assert_close(
            policy.calls[2]["memory_state"].working, after_permute.working
        )

    def test_noreset_preserves_state_across_resets(self):
        policy = _RecordingPolicy()
        server = WebsocketPolicyServer(policy, memory_config=MemoryServerConfig(mode="noreset"))
        session = _ConnectionSession()
        _reset(server, session)
        _infer(server, session)
        carried = session.memory_state
        self.assertTrue(_reset(server, session, seed=8)["ok"])
        self.assertIs(session.memory_state, carried)
        self.assertEqual(session.decision_index, 0)
        _infer(server, session)
        self.assertIs(policy.calls[-1]["memory_state"], carried)


if __name__ == "__main__":
    unittest.main()
