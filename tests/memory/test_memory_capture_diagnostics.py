import unittest

import torch

from starVLA.model.modules.memory import RecurrentMemory


class CaptureDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.memory = RecurrentMemory(
            source_dim=12,
            memory_dim=8,
            num_slots=3,
            num_heads=2,
            update_gate_init=0.1,
        )
        self.memory.eval()
        self.state = self.memory.init_state(2, torch.device("cpu"))
        self.source = torch.randn(2, 5, 12)

    def _pre_change_write_working(self):
        """Replicate the original write arithmetic exactly (need_weights=False)."""
        with torch.no_grad():
            source = self.memory.source_projection(self.memory.source_norm(self.source))
            previous = self.state.working
            query = self.memory.slot_norm(previous + self.memory.slot_ids.unsqueeze(0))
            context, _ = self.memory.update_attention(
                query=query, key=source, value=source, need_weights=False
            )
            gate = torch.sigmoid(
                self.memory.update_gate(
                    torch.cat((self.memory.slot_norm(previous), context), dim=-1)
                )
            )
            candidate = torch.tanh(self.memory.candidate_projection(context))
            return (1.0 - gate) * previous + gate * candidate

    def test_capture_off_is_bit_identical_and_attributes_none(self):
        self.assertFalse(self.memory.capture_diagnostics)
        with torch.no_grad():
            read = self.memory.read(self.source, self.state)
            written = self.memory.write(self.source, self.state)
        self.assertTrue(torch.equal(read.tokens, self.state.working))
        self.assertTrue(torch.equal(written.working, self._pre_change_write_working()))
        self.assertIsNone(self.memory.last_read_diagnostics)
        self.assertIsNone(self.memory.last_write_diagnostics)

    def test_capture_on_populates_pinned_keys_with_correct_shapes(self):
        self.memory.capture_diagnostics = True
        self.memory.read(self.source, self.state)
        self.memory.write(self.source, self.state)

        self.assertEqual(set(self.memory.last_read_diagnostics), {"read_attention"})

        write_diag = self.memory.last_write_diagnostics
        self.assertEqual(
            set(write_diag),
            {
                "update_gate_mean",
                "update_gate_p05",
                "update_gate_p95",
                "per_slot_delta_norm",
                "slot_cosine_mean",
                "write_attention",
            },
        )
        for key in ("update_gate_mean", "update_gate_p05", "update_gate_p95", "slot_cosine_mean"):
            self.assertIsInstance(write_diag[key], float)
        # Zero-initialized gate weights make the update gate exactly sigmoid(bias).
        self.assertAlmostEqual(write_diag["update_gate_mean"], 0.1, places=5)
        self.assertLessEqual(write_diag["update_gate_p05"], write_diag["update_gate_p95"])

        self.assertEqual(write_diag["per_slot_delta_norm"].shape, (2, 3))
        self.assertFalse(write_diag["per_slot_delta_norm"].requires_grad)
        self.assertEqual(write_diag["write_attention"].shape, (2, 3, 5))
        self.assertFalse(write_diag["write_attention"].requires_grad)
        torch.testing.assert_close(
            write_diag["write_attention"].sum(dim=-1), torch.ones(2, 3)
        )


if __name__ == "__main__":
    unittest.main()
