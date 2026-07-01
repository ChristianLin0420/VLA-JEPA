import unittest

import torch

from starVLA.model.modules.memory import RecurrentMemory, ResidualMemoryFusion


class ResidualMemoryFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_zero_gate_is_exact_functional_noop(self):
        fusion = ResidualMemoryFusion(
            consumer_dim=12,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
            gate_init=0.0,
        )
        consumer = torch.randn(2, 5, 12)
        memory = torch.randn(2, 3, 8)
        output = fusion(consumer, memory)
        self.assertTrue(torch.equal(output, consumer))

        bf16_consumer = consumer.to(dtype=torch.bfloat16)
        bf16_output = fusion(bf16_consumer, memory)
        self.assertEqual(bf16_output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(bf16_output, bf16_consumer))

    def test_zero_gate_first_gradient_reaches_gate_only(self):
        fusion = ResidualMemoryFusion(
            consumer_dim=12,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
            gate_init=0.0,
        )
        consumer = torch.randn(2, 5, 12)
        memory = torch.randn(2, 3, 8)
        fusion(consumer, memory).square().mean().backward()

        self.assertIsNotNone(fusion.gate.grad)
        self.assertGreater(float(fusion.gate.grad.abs()), 0.0)
        adapter_grad = fusion.query_projection.weight.grad
        self.assertIsNotNone(adapter_grad)
        self.assertEqual(float(adapter_grad.abs().sum()), 0.0)

    def test_small_gate_trains_adapter_and_preserves_shape_dtype(self):
        fusion = ResidualMemoryFusion(
            consumer_dim=12,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
            gate_init=1.0e-3,
        )
        consumer = torch.randn(2, 5, 12, dtype=torch.float32)
        memory = torch.randn(2, 3, 8, dtype=torch.float32)
        output = fusion(consumer, memory)
        self.assertEqual(output.shape, consumer.shape)
        self.assertEqual(output.dtype, consumer.dtype)
        self.assertFalse(torch.equal(output, consumer))
        output.square().mean().backward()
        self.assertGreater(float(fusion.query_projection.weight.grad.abs().sum()), 0.0)

    def test_bypass_and_validation(self):
        fusion = ResidualMemoryFusion(
            consumer_dim=12,
            memory_dim=8,
            bottleneck_dim=8,
            num_heads=2,
        )
        consumer = torch.randn(2, 5, 12)
        memory = torch.randn(2, 3, 8)
        self.assertIs(fusion(consumer, memory, bypass=True), consumer)
        with self.assertRaises(ValueError):
            fusion(consumer, torch.randn(3, 3, 8))
        with self.assertRaises(ValueError):
            fusion(consumer, torch.randn(2, 3, 7))

    def test_default_phase_one_package_stays_below_parameter_cap(self):
        memory = RecurrentMemory()
        fusion = ResidualMemoryFusion()
        parameter_count = sum(p.numel() for p in memory.parameters())
        parameter_count += sum(p.numel() for p in fusion.parameters())
        self.assertLess(parameter_count, 10_000_000)
        self.assertGreater(float(memory.source_projection.weight.std()), 0.0)
        self.assertGreater(float(fusion.output_projection.weight.std()), 0.0)


if __name__ == "__main__":
    unittest.main()
