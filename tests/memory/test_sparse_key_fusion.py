import math
import unittest

import torch
import torch.nn.functional as F

from starVLA.model.modules.memory import (
    MemoryState,
    RecurrentMemory,
    SparseKeyMemoryFusion,
)

CONSUMER_DIM = 16
MEMORY_DIM = 8
KEY_DIM = 6
NUM_SLOTS = 4
NUM_TOKENS = 32


def _make_state(batch=2, working_scale=1.0):
    torch.manual_seed(13)
    return MemoryState(
        working=torch.randn(batch, NUM_SLOTS, MEMORY_DIM) * working_scale,
        episodic=None,
        steps=torch.tensor([3, 11], dtype=torch.int64)[:batch],
        valid=torch.ones(batch, dtype=torch.bool),
        keys=torch.randn(batch, NUM_SLOTS, KEY_DIM),
    )


class SparseKeyMemoryFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.fusion = SparseKeyMemoryFusion(
            consumer_dim=CONSUMER_DIM,
            memory_dim=MEMORY_DIM,
            key_dim=KEY_DIM,
            num_slots=NUM_SLOTS,
        )
        self.consumer = torch.randn(2, NUM_TOKENS, CONSUMER_DIM)
        self.state = _make_state()

    def test_output_shape_appends_one_tap_token(self):
        output = self.fusion(self.consumer, self.state)
        self.assertEqual(output.shape, (2, NUM_TOKENS + 1, CONSUMER_DIM))
        self.assertEqual(output.dtype, self.consumer.dtype)

        bf16_output = self.fusion(self.consumer.to(dtype=torch.bfloat16), self.state)
        self.assertEqual(bf16_output.dtype, torch.bfloat16)
        self.assertEqual(bf16_output.shape, (2, NUM_TOKENS + 1, CONSUMER_DIM))

    def test_zero_gamma_kills_content_exactly_but_keeps_the_tap(self):
        self.assertEqual(float(self.fusion.gamma_c), 0.0)
        output = self.fusion(self.consumer, self.state)
        self.assertTrue(torch.equal(output[:, :NUM_TOKENS], self.consumer))
        self.assertGreater(float(output[:, NUM_TOKENS].norm()), 0.0)

    def test_zero_gamma_first_gradient_reaches_gamma_and_tap_only(self):
        self.fusion(self.consumer, self.state).square().mean().backward()
        self.assertGreater(float(self.fusion.gamma_c.grad.abs()), 0.0)
        self.assertGreater(float(self.fusion.time_mlp[0].weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(self.fusion.qk_proj.weight.grad.abs().sum()), 0.0)

    def test_bypass_is_shape_stable_with_a_zero_tap(self):
        output = self.fusion(self.consumer, self.state, bypass=True)
        self.assertEqual(output.shape, (2, NUM_TOKENS + 1, CONSUMER_DIM))
        self.assertTrue(torch.equal(output[:, :NUM_TOKENS], self.consumer))
        self.assertEqual(torch.count_nonzero(output[:, NUM_TOKENS]).item(), 0)
        self.assertIsNone(self.fusion.last_residual)

    def test_whitening_kills_norm_information(self):
        scaled = _make_state(working_scale=10.0)
        with torch.no_grad():
            attention, _ = self.fusion._sparse_attention(self.consumer, self.state.keys)
            residuals = [
                self.fusion.out_proj(
                    torch.matmul(attention, self.fusion.value_proj(state.working))
                )
                for state in (self.state, scaled)
            ]
            norms = [F.layer_norm(r, (CONSUMER_DIM,)).norm(dim=-1) for r in residuals]
        self.assertGreater(float(residuals[1].norm() / residuals[0].norm()), 5.0)
        expected = torch.full_like(norms[0], math.sqrt(CONSUMER_DIM))
        torch.testing.assert_close(norms[0], expected, atol=1.0e-2, rtol=0.0)
        torch.testing.assert_close(norms[1], expected, atol=1.0e-2, rtol=0.0)
        torch.testing.assert_close(norms[0], norms[1], atol=1.0e-3, rtol=0.0)

    def test_attention_is_exactly_top_two_sparse(self):
        with torch.no_grad():
            attention, _ = self.fusion._sparse_attention(self.consumer, self.state.keys)
        self.assertEqual(attention.shape, (2, NUM_TOKENS, NUM_SLOTS))
        nonzero = attention.ne(0.0).sum(dim=-1)
        self.assertTrue(nonzero.eq(2).all().item())
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, NUM_TOKENS))

    def test_pre_gate_residual_is_captured_live(self):
        self.fusion(self.consumer, self.state)
        residual = self.fusion.last_residual
        self.assertEqual(residual.shape, (2, NUM_TOKENS, CONSUMER_DIM))
        self.assertEqual(residual.dtype, torch.float32)
        self.assertTrue(residual.requires_grad)

    def test_diagnostics_only_under_capture(self):
        self.fusion(self.consumer, self.state)
        self.assertIsNone(self.fusion.last_fusion_diagnostics)

        self.fusion.capture_diagnostics = True
        with torch.no_grad():
            self.fusion(self.consumer, self.state)
        diagnostics = self.fusion.last_fusion_diagnostics
        self.assertEqual(set(diagnostics), {"injection_ratio", "match_margin", "tap_norm"})
        for value in diagnostics.values():
            self.assertIsInstance(value, float)
            self.assertTrue(math.isfinite(value))
        self.assertGreater(diagnostics["tap_norm"], 0.0)

        with torch.no_grad():
            self.fusion(self.consumer, self.state, bypass=True)
        self.assertEqual(
            self.fusion.last_fusion_diagnostics,
            {"injection_ratio": 0.0, "match_margin": 0.0, "tap_norm": 0.0},
        )

    def test_residual_scale_zero_removes_content_with_a_live_gamma(self):
        with torch.no_grad():
            self.fusion.gamma_c.fill_(1.0)
        self.fusion.residual_scale = 0.0
        with torch.no_grad():
            output = self.fusion(self.consumer, self.state)
        self.assertTrue(torch.equal(output[:, :NUM_TOKENS], self.consumer))

    def test_validation_and_defaults_stay_below_parameter_cap(self):
        with self.assertRaises(ValueError):
            SparseKeyMemoryFusion(consumer_dim=CONSUMER_DIM, num_slots=1)
        with self.assertRaises(ValueError):
            self.fusion(torch.randn(2, 4, CONSUMER_DIM + 1), self.state)
        schema_one = MemoryState(
            working=self.state.working,
            episodic=None,
            steps=self.state.steps,
            valid=self.state.valid,
        )
        with self.assertRaises(ValueError):
            self.fusion(self.consumer, schema_one)

        memory = RecurrentMemory(use_keys=True)
        fusion = SparseKeyMemoryFusion()
        parameter_count = sum(p.numel() for p in memory.parameters())
        parameter_count += sum(p.numel() for p in fusion.parameters())
        self.assertLess(parameter_count, 10_000_000)


if __name__ == "__main__":
    unittest.main()
