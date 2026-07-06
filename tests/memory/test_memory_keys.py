import unittest

import torch

from starVLA.model.modules.memory import RecurrentMemory


def _make_memory(use_keys, seed=7):
    torch.manual_seed(seed)
    return RecurrentMemory(
        source_dim=12,
        memory_dim=8,
        num_slots=3,
        num_heads=2,
        update_gate_init=0.1,
        key_dim=4,
        use_keys=use_keys,
    )


class RecurrentMemoryKeysTest(unittest.TestCase):
    def setUp(self):
        self.memory = _make_memory(use_keys=True)
        self.state = self.memory.init_state(2, torch.device("cpu"))
        torch.manual_seed(23)
        self.source = torch.randn(2, 5, 12)

    def _expected_write(self, source, state):
        """Replicate write arithmetic exactly (need_weights=False)."""
        with torch.no_grad():
            m = self.memory
            projected = m.source_projection(m.source_norm(source))
            previous = state.working
            query = m.slot_norm(previous + m.slot_ids.unsqueeze(0))
            context, _ = m.update_attention(
                query=query, key=projected, value=projected, need_weights=False
            )
            gate = torch.sigmoid(
                m.update_gate(torch.cat((m.slot_norm(previous), context), dim=-1))
            )
            gate_mean = gate.mean(dim=-1, keepdim=True)
            keys = (1.0 - gate_mean) * state.keys + gate_mean * torch.tanh(
                m.key_projection(context)
            )
            return keys

    def test_init_and_reset_fill_keys_from_learned_initial_keys(self):
        state = self.memory.init_state(
            2, torch.device("cpu"), valid_mask=torch.tensor([True, False])
        )
        self.assertEqual(state.keys.shape, (2, 3, 4))
        self.assertEqual(state.keys.dtype, torch.float32)
        self.assertTrue(torch.equal(state.keys[0], self.memory.initial_keys.detach()))
        self.assertEqual(torch.count_nonzero(state.keys[1]).item(), 0)

        written = self.memory.write(self.source, self.state)
        reset = self.memory.reset_state(written, torch.tensor([True, False]))
        self.assertTrue(torch.equal(reset.keys[0], self.memory.initial_keys.detach()))
        self.assertTrue(torch.equal(reset.keys[1], written.keys[1]))

    def test_keys_follow_the_write_gate_and_convex_update_rule(self):
        with torch.no_grad():
            written = self.memory.write(self.source, self.state)
        self.assertTrue(torch.equal(written.keys, self._expected_write(self.source, self.state)))

        # gate -> 0: keys stay at their previous values.
        with torch.no_grad():
            self.memory.update_gate.bias.fill_(-20.0)
            frozen = self.memory.write(self.source, self.state)
        torch.testing.assert_close(frozen.keys, self.state.keys, atol=1.0e-6, rtol=0.0)

        # gate -> 1: keys become the tanh-bounded candidate from the write context.
        with torch.no_grad():
            self.memory.update_gate.bias.fill_(20.0)
            replaced = self.memory.write(self.source, self.state)
        self.assertTrue(replaced.keys.abs().lt(1.0).all().item())
        self.assertFalse(torch.allclose(replaced.keys, self.state.keys))

    def test_keys_stay_bounded_across_repeated_writes(self):
        state = self.state
        for _ in range(6):
            state = self.memory.write(torch.randn(2, 5, 12), state)
        self.assertTrue(state.keys.abs().lt(1.0).all().item())

    def test_update_mask_freezes_keys_per_row(self):
        written = self.memory.write(
            self.source, self.state, update_mask=torch.tensor([True, False])
        )
        self.assertFalse(torch.equal(written.keys[0], self.state.keys[0]))
        self.assertTrue(torch.equal(written.keys[1], self.state.keys[1]))

    def test_count_mask_ticks_independently_of_updates(self):
        written = self.memory.write(
            self.source,
            self.state,
            update_mask=torch.tensor([True, False]),
            count_mask=torch.tensor([False, True]),
        )
        self.assertEqual(written.steps.tolist(), [0, 1])
        self.assertFalse(torch.equal(written.working[0], self.state.working[0]))
        self.assertTrue(torch.equal(written.working[1], self.state.working[1]))
        self.assertTrue(torch.equal(written.keys[1], self.state.keys[1]))

        # Default count semantics remain bound to the update mask (memv1).
        default = self.memory.write(
            self.source, self.state, update_mask=torch.tensor([True, False])
        )
        self.assertEqual(default.steps.tolist(), [1, 0])

    def test_use_keys_false_is_bit_identical_to_schema_one(self):
        base = _make_memory(use_keys=False, seed=5)
        keyed = _make_memory(use_keys=True, seed=5)
        self.assertIsNone(base.initial_keys)
        self.assertIsNone(base.key_projection)
        state_keys = set(base.state_dict())
        self.assertNotIn("initial_keys", state_keys)
        self.assertFalse(any(key.startswith("key_projection") for key in state_keys))

        torch.manual_seed(29)
        source = torch.randn(2, 5, 12)
        base_state = base.init_state(2, torch.device("cpu"))
        keyed_state = keyed.init_state(2, torch.device("cpu"))
        self.assertIsNone(base_state.keys)
        self.assertIsNotNone(keyed_state.keys)

        with torch.no_grad():
            base_written = base.write(source, base_state)
            keyed_written = keyed.write(source, keyed_state)
        self.assertTrue(torch.equal(base_written.working, keyed_written.working))
        self.assertTrue(torch.equal(base_written.steps, keyed_written.steps))
        self.assertIsNone(base_written.keys)

    def test_missing_keys_are_rejected_when_use_keys_is_set(self):
        base = _make_memory(use_keys=False, seed=5)
        schema_one_state = base.init_state(2, torch.device("cpu"))
        with self.assertRaises(ValueError):
            self.memory.write(self.source, schema_one_state)
        with self.assertRaises(TypeError):
            self.memory.write(
                self.source,
                self.state,
                count_mask=torch.ones(2, dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
