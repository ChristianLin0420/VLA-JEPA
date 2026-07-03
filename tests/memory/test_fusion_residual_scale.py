import math
import unittest

import torch

from starVLA.model.modules.memory import ResidualMemoryFusion


class ResidualScaleTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.fusion = ResidualMemoryFusion(
            consumer_dim=12,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
            gate_init=1.0e-3,
        )
        self.fusion.eval()
        self.consumer = torch.randn(2, 5, 12)
        self.memory = torch.randn(2, 3, 8)

    def _pre_change_output(self):
        """Replicate the original forward exactly: consumer + tanh(gate)·residual."""
        with torch.no_grad():
            consumer = self.consumer.to(dtype=torch.float32)
            memory = self.memory.to(dtype=torch.float32)
            query = self.fusion.query_projection(self.fusion.consumer_norm(consumer))
            key_value = self.fusion.memory_projection(self.fusion.memory_norm(memory))
            attended, _ = self.fusion.attention(
                query=query, key=key_value, value=key_value, need_weights=False
            )
            residual = self.fusion.output_projection(attended)
            return self.consumer + (torch.tanh(self.fusion.gate) * residual).to(
                dtype=self.consumer.dtype
            )

    def test_default_scale_is_bit_identical_to_pre_change_path(self):
        self.assertEqual(self.fusion.residual_scale, 1.0)
        with torch.no_grad():
            output = self.fusion(self.consumer, self.memory)
        self.assertTrue(torch.equal(output, self._pre_change_output()))

    def test_zero_scale_equals_bypass_bit_exactly(self):
        self.fusion.residual_scale = 0.0
        with torch.no_grad():
            scaled = self.fusion(self.consumer, self.memory)
            bypassed = self.fusion(self.consumer, self.memory, bypass=True)
        self.assertTrue(torch.equal(scaled, bypassed))

    def test_injection_ratio_is_finite_float_and_linear_in_scale(self):
        self.fusion.capture_diagnostics = True
        ratios = {}
        for scale in (0.5, 1.0, 2.0):
            self.fusion.residual_scale = scale
            with torch.no_grad():
                self.fusion(self.consumer, self.memory)
            ratio = self.fusion.last_fusion_diagnostics["injection_ratio"]
            self.assertIsInstance(ratio, float)
            self.assertTrue(math.isfinite(ratio))
            self.assertGreater(ratio, 0.0)
            ratios[scale] = ratio
        self.assertAlmostEqual(ratios[1.0] / ratios[0.5], 2.0, places=5)
        self.assertAlmostEqual(ratios[2.0] / ratios[1.0], 2.0, places=5)

    def test_capture_off_leaves_diagnostics_none(self):
        # The default training path performs no diagnostic arithmetic: both
        # the fused and the bypass forward must leave the side-channel empty.
        self.assertFalse(self.fusion.capture_diagnostics)
        with torch.no_grad():
            self.fusion(self.consumer, self.memory)
        self.assertIsNone(self.fusion.last_fusion_diagnostics)
        with torch.no_grad():
            self.fusion(self.consumer, self.memory, bypass=True)
        self.assertIsNone(self.fusion.last_fusion_diagnostics)

    def test_bypass_reports_zero_injection_ratio_under_capture(self):
        self.fusion.capture_diagnostics = True
        with torch.no_grad():
            self.fusion(self.consumer, self.memory, bypass=True)
        self.assertEqual(self.fusion.last_fusion_diagnostics, {"injection_ratio": 0.0})

    def test_capture_records_read_attention_map(self):
        self.fusion.capture_diagnostics = True
        with torch.no_grad():
            self.fusion(self.consumer, self.memory)
        attention = self.fusion.last_fusion_diagnostics["read_attention"]
        self.assertEqual(attention.shape, (2, 5, 3))
        torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 5))


if __name__ == "__main__":
    unittest.main()
