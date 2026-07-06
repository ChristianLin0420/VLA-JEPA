import unittest

import torch

from starVLA.model.modules.memory import MemoryRead, MemoryState


class MemoryStateTest(unittest.TestCase):
    def test_contract_detach_and_device_move_preserve_dtypes(self):
        working = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
        episodic = torch.randn(2, 5, 6, dtype=torch.float32, requires_grad=True)
        state = MemoryState(
            working=working,
            episodic=episodic,
            steps=torch.tensor([1, 2], dtype=torch.int64),
            valid=torch.tensor([True, False]),
        )

        detached = state.detach()
        self.assertFalse(detached.working.requires_grad)
        self.assertFalse(detached.episodic.requires_grad)
        self.assertEqual(detached.working.dtype, torch.float32)
        self.assertEqual(detached.steps.dtype, torch.int64)
        self.assertEqual(detached.valid.dtype, torch.bool)

        moved = detached.to(device=torch.device("cpu"))
        self.assertEqual(moved.working.dtype, torch.float32)
        self.assertEqual(moved.episodic.dtype, torch.float32)
        self.assertEqual(moved.steps.dtype, torch.int64)
        self.assertEqual(moved.valid.dtype, torch.bool)
        self.assertEqual(moved.batch_size, 2)

    def test_rejects_invalid_shape_dtype_and_device_contracts(self):
        with self.assertRaises(TypeError):
            MemoryState(
                working=torch.zeros(1, 2, 3, dtype=torch.bfloat16),
                episodic=None,
                steps=torch.zeros(1, dtype=torch.int64),
                valid=torch.ones(1, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            MemoryState(
                working=torch.zeros(2, 2, 3),
                episodic=None,
                steps=torch.zeros(1, dtype=torch.int64),
                valid=torch.ones(2, dtype=torch.bool),
            )
        with self.assertRaises(TypeError):
            MemoryState(
                working=torch.zeros(1, 2, 3),
                episodic=None,
                steps=torch.zeros(1, dtype=torch.int32),
                valid=torch.ones(1, dtype=torch.bool),
            )

    def test_keys_default_to_none_and_round_trip_detach_and_move(self):
        schema_one = MemoryState(
            working=torch.zeros(2, 3, 4),
            episodic=None,
            steps=torch.zeros(2, dtype=torch.int64),
            valid=torch.ones(2, dtype=torch.bool),
        )
        self.assertIsNone(schema_one.keys)
        self.assertIsNone(schema_one.detach().keys)
        self.assertIsNone(schema_one.to(device=torch.device("cpu")).keys)

        state = MemoryState(
            working=torch.zeros(2, 3, 4),
            episodic=None,
            steps=torch.zeros(2, dtype=torch.int64),
            valid=torch.ones(2, dtype=torch.bool),
            keys=torch.randn(2, 3, 5, requires_grad=True),
        )
        detached = state.detach()
        self.assertFalse(detached.keys.requires_grad)
        moved = detached.to(device=torch.device("cpu"))
        self.assertEqual(moved.keys.dtype, torch.float32)
        self.assertEqual(moved.keys.shape, (2, 3, 5))

    def test_keys_validation_mirrors_episodic_contract(self):
        working = torch.zeros(2, 3, 4)
        steps = torch.zeros(2, dtype=torch.int64)
        valid = torch.ones(2, dtype=torch.bool)
        with self.assertRaises(TypeError):
            MemoryState(
                working=working,
                episodic=None,
                steps=steps,
                valid=valid,
                keys=torch.zeros(2, 3, 5, dtype=torch.bfloat16),
            )
        with self.assertRaises(ValueError):
            MemoryState(
                working=working,
                episodic=None,
                steps=steps,
                valid=valid,
                keys=torch.zeros(1, 3, 5),
            )
        with self.assertRaises(ValueError):
            MemoryState(
                working=working,
                episodic=None,
                steps=steps,
                valid=valid,
                keys=torch.zeros(2, 3),
            )

    def test_memory_read_contract(self):
        read = MemoryRead(
            tokens=torch.zeros(2, 3, 4, dtype=torch.float32),
            diagnostics={"norm": torch.zeros(2)},
        )
        self.assertEqual(read.tokens.shape, (2, 3, 4))
        with self.assertRaises(TypeError):
            MemoryRead(tokens=torch.zeros(2, 3, 4, dtype=torch.bfloat16), diagnostics={})


if __name__ == "__main__":
    unittest.main()
