import unittest

import torch

from starVLA.model.modules.memory import MemoryState, RecurrentMemory, ResidualMemoryFusion


class RecurrentMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.memory = RecurrentMemory(
            source_dim=12,
            memory_dim=8,
            num_slots=3,
            num_heads=2,
            update_gate_init=0.1,
        )

    def test_init_read_write_shapes_masks_and_fp32(self):
        valid = torch.tensor([True, False])
        state = self.memory.init_state(2, torch.device("cpu"), valid_mask=valid)
        source = torch.randn(2, 5, 12, dtype=torch.bfloat16)

        read = self.memory.read(source, state)
        self.assertEqual(read.tokens.shape, (2, 3, 8))
        self.assertEqual(read.tokens.dtype, torch.float32)
        self.assertTrue(torch.count_nonzero(read.tokens[1]).item() == 0)

        before = state.working.clone()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            after = self.memory.write(source, state, update_mask=torch.tensor([True, True]))

        self.assertEqual(after.working.dtype, torch.float32)
        self.assertEqual(after.steps.tolist(), [1, 0])
        self.assertTrue(torch.equal(state.working, before))
        self.assertTrue(torch.equal(after.working[1], state.working[1]))
        self.assertNotEqual(after.working.data_ptr(), state.working.data_ptr())

    def test_reset_is_selective_non_aliasing_and_can_activate_a_row(self):
        state = self.memory.init_state(2, torch.device("cpu"))
        source = torch.randn(2, 4, 12)
        changed = self.memory.write(source, state)
        reset = self.memory.reset_state(changed, torch.tensor([True, False]))
        fresh = self.memory.init_state(2, torch.device("cpu"))

        self.assertTrue(torch.equal(reset.working[0], fresh.working[0]))
        self.assertTrue(torch.equal(reset.working[1], changed.working[1]))
        self.assertEqual(reset.steps.tolist(), [0, 1])
        self.assertNotEqual(reset.working.data_ptr(), changed.working.data_ptr())

        inactive = self.memory.init_state(
            2, torch.device("cpu"), valid_mask=torch.tensor([True, False])
        )
        activated = self.memory.reset_state(inactive, torch.tensor([False, True]))
        self.assertEqual(activated.valid.tolist(), [True, True])
        self.assertTrue(torch.equal(activated.working[1], fresh.working[1]))

    def test_valid_mask_zeros_and_preserves_optional_episodic_state(self):
        base = self.memory.init_state(2, torch.device("cpu"))
        state = MemoryState(
            working=base.working,
            episodic=torch.ones(2, 4, 4, dtype=torch.float32),
            steps=torch.tensor([3, 4], dtype=torch.int64),
            valid=torch.tensor([True, True]),
        )
        reset = self.memory.reset_state(
            state,
            reset_mask=torch.tensor([True, False]),
            valid_mask=torch.tensor([True, False]),
        )
        self.assertEqual(reset.steps.tolist(), [0, 0])
        self.assertTrue(torch.count_nonzero(reset.episodic).item() == 0)
        self.assertTrue(torch.count_nonzero(reset.working[1]).item() == 0)

    def test_read_before_write_and_runtime_state_is_not_serialized(self):
        state = self.memory.init_state(1, torch.device("cpu"))
        source = torch.randn(1, 4, 12)
        before = self.memory.read(source, state)
        after_state = self.memory.write(source, state)
        after = self.memory.read(source, after_state)

        self.assertTrue(torch.equal(before.tokens, state.working))
        self.assertTrue(torch.equal(after.tokens, after_state.working))
        self.assertFalse(torch.equal(before.tokens, after.tokens))
        self.assertFalse(hasattr(self.memory, "memory_state"))
        keys = set(self.memory.state_dict())
        self.assertFalse(any("steps" in key or "valid" in key or "episodic" in key for key in keys))

    def test_delayed_gradient_reaches_earlier_writer(self):
        fusion = ResidualMemoryFusion(
            consumer_dim=10,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
            gate_init=0.1,
        )
        state0 = self.memory.init_state(2, torch.device("cpu"))
        source0 = torch.randn(2, 4, 12)
        state1 = self.memory.write(source0, state0)
        source1 = torch.randn(2, 4, 12)
        read1 = self.memory.read(source1, state1)
        consumer1 = torch.randn(2, 5, 10)
        loss = fusion(consumer1, read1.tokens).square().mean()
        loss.backward()

        grad = self.memory.source_projection.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all().item())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_shape_and_mask_errors_are_explicit(self):
        state = self.memory.init_state(2, torch.device("cpu"))
        with self.assertRaises(ValueError):
            self.memory.write(torch.randn(2, 4, 11), state)
        with self.assertRaises(TypeError):
            self.memory.read(
                torch.randn(2, 4, 12), state, read_mask=torch.ones(2, dtype=torch.int64)
            )
        with self.assertRaises(ValueError):
            self.memory.reset_state(state, torch.ones(3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
