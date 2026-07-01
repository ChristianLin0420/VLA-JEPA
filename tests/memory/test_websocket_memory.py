import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn

from deployment.model_server.tools.websocket_policy_server import (
    WebsocketPolicyServer,
    _ConnectionSession,
)


class _FakeMemoryPolicy(nn.Module):
    memory_enabled = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def predict_action(
        self,
        *,
        memory_state=None,
        generator=None,
        return_memory_state=False,
        update_memory=True,
        fail=False,
        **kwargs,
    ):
        value = torch.randn((), generator=generator).item()
        if fail:
            raise RuntimeError("synthetic failure")
        previous = 0 if memory_state is None else int(memory_state)
        candidate = previous + 1 if update_memory else previous
        output = {"normalized_actions": np.asarray([[[value]]], dtype=np.float32)}
        return (output, candidate) if return_memory_state else output


class _FakeStatelessPolicy(nn.Module):
    memory_enabled = False

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def predict_action(self, *, batch_images, **kwargs):
        return {"normalized_actions": np.zeros((len(batch_images), 1, 1), dtype=np.float32)}


def _reset(server, session, seed=7):
    return server._route_message(
        {
            "type": "reset",
            "request_id": "reset",
            "payload": {"episode_id": "episode", "episode_seed": seed},
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


class WebsocketMemoryTest(unittest.TestCase):
    def test_reset_is_required_and_sessions_are_isolated(self):
        server = WebsocketPolicyServer(_FakeMemoryPolicy())
        first = _ConnectionSession()
        second = _ConnectionSession()
        self.assertFalse(_infer(server, first)["ok"])
        self.assertTrue(_reset(server, first)["ok"])
        self.assertTrue(_reset(server, second)["ok"])

        out_first = _infer(server, first)
        out_second = _infer(server, second)
        self.assertTrue(out_first["ok"])
        self.assertTrue(out_second["ok"])
        np.testing.assert_array_equal(
            out_first["data"]["normalized_actions"],
            out_second["data"]["normalized_actions"],
        )
        self.assertEqual(first.memory_state, 1)
        self.assertEqual(second.memory_state, 1)

        _infer(server, first)
        self.assertEqual(first.memory_state, 2)
        self.assertEqual(second.memory_state, 1)

    def test_failure_rolls_back_memory_and_rng(self):
        server = WebsocketPolicyServer(_FakeMemoryPolicy())
        failed = _ConnectionSession()
        control = _ConnectionSession()
        _reset(server, failed, seed=11)
        _reset(server, control, seed=11)

        error = _infer(server, failed, fail=True)
        self.assertFalse(error["ok"])
        self.assertIsNone(failed.memory_state)
        retry = _infer(server, failed)
        expected = _infer(server, control)
        np.testing.assert_array_equal(
            retry["data"]["normalized_actions"],
            expected["data"]["normalized_actions"],
        )
        self.assertEqual(failed.memory_state, control.memory_state)

    def test_zero_mode_never_commits_state(self):
        server = WebsocketPolicyServer(_FakeMemoryPolicy(), memory_mode="zero")
        session = _ConnectionSession()
        _reset(server, session)
        self.assertTrue(_infer(server, session)["ok"])
        self.assertTrue(_infer(server, session)["ok"])
        self.assertIsNone(session.memory_state)

    def test_failed_reset_and_legacy_reset_are_transactional(self):
        server = WebsocketPolicyServer(_FakeMemoryPolicy())
        session = _ConnectionSession()
        self.assertTrue(_reset(server, session, seed=5)["ok"])
        _infer(server, session)
        old_state = session.memory_state
        old_rng = session.generator.get_state().clone()
        rejected = server._route_message(
            {
                "type": "reset",
                "payload": {"episode_id": "bad", "episode_seed": 2**100},
            },
            session=session,
        )
        self.assertFalse(rejected["ok"])
        self.assertEqual(session.memory_state, old_state)
        torch.testing.assert_close(session.generator.get_state(), old_rng)

        legacy = server._route_message(
            {"instruction": "legacy episode", "reset": True}, session=session
        )
        self.assertTrue(legacy["ok"])
        self.assertEqual(legacy["type"], "reset_result")
        self.assertIsNone(session.memory_state)

    def test_stateless_policy_may_change_batch_size(self):
        server = WebsocketPolicyServer(_FakeStatelessPolicy())
        session = _ConnectionSession()
        one = _infer(server, session)
        self.assertTrue(one["ok"])
        payload = {
            "batch_images": [
                [Image.new("RGB", (2, 2))],
                [Image.new("RGB", (2, 2))],
            ],
            "instructions": ["a", "b"],
        }
        two = server._route_message(
            {"type": "infer", "payload": payload}, session=session
        )
        self.assertTrue(two["ok"])


if __name__ == "__main__":
    unittest.main()
