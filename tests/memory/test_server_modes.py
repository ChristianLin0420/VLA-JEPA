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


def _reset(server, session, seed=7, episode_id="episode"):
    return server._route_message(
        {
            "type": "reset",
            "request_id": "reset",
            "payload": {"episode_id": episode_id, "episode_seed": seed},
        },
        session=session,
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

    def test_permute_schedule(self):
        for plan in _schedule(MemoryServerConfig(mode="permute")):
            self.assertEqual(
                (plan.memory_state_override, plan.update_memory, plan.memory_bypass),
                ("permute", True, False),
            )

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
            _reset(server, session, episode_id="libero_10-task3-ep0")
            for _ in range(3):
                self.assertTrue(_infer(server, session)["ok"])
            episode_dir = Path(tmp) / "libero_10-task3-ep0"
            self.assertEqual(
                sorted(p.name for p in episode_dir.iterdir()), ["d0.pt", "d1.pt", "d2.pt"]
            )

            donor_policy = _RecordingPolicy()
            donor_config = MemoryServerConfig(mode="foreign", donor_dir=tmp)
            donor_server = WebsocketPolicyServer(donor_policy, memory_config=donor_config)
            donor_session = _ConnectionSession()
            _reset(donor_server, donor_session)
            for _ in range(4):
                self.assertTrue(_infer(donor_server, donor_session)["ok"])
            # decision-index-matched donor states, clamped at the donor's last decision
            injected = [float(call["memory_state"].working[0, 0, 0]) for call in donor_policy.calls]
            self.assertEqual(injected, [1.0, 2.0, 3.0, 3.0])
            self.assertFalse(any(call["update_memory"] for call in donor_policy.calls))
            self.assertIsNone(donor_session.memory_state)

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

    def test_permute_applies_fixed_slot_permutation_to_live_state(self):
        state = _make_state(0.0, slots=8, dim=2)
        state.working.copy_(torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 2))
        permuted = _permute_slots(state)
        order = permuted.working[0, :, 0]
        self.assertFalse(torch.equal(order, state.working[0, :, 0]))
        torch.testing.assert_close(order.sort().values, state.working[0, :, 0])
        torch.testing.assert_close(_permute_slots(state).working, permuted.working)

        policy = _RecordingPolicy()
        server = WebsocketPolicyServer(policy, memory_config=MemoryServerConfig(mode="permute"))
        session = _ConnectionSession()
        _reset(server, session)
        _infer(server, session)
        self.assertIsNone(policy.calls[0]["memory_state"])  # nothing to permute yet
        committed = session.memory_state
        _infer(server, session)
        torch.testing.assert_close(
            policy.calls[1]["memory_state"].working, _permute_slots(committed).working
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
