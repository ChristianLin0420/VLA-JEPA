"""memv3 Retro-JEPA Stage M1: masked-past retrodiction unit tests (CPU)."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from starVLA.model.framework.VLA_JEPA import VLA_JEPA
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC

VJ_HIDDEN = 8
TEACHER = 2 * VJ_HIDDEN
QWEN_HIDDEN = 12
TOKENS_PER_FRAME = 8


class _VjProcessorStub:
    def __call__(self, videos, return_tensors):
        clip = torch.as_tensor(np.asarray(videos), dtype=torch.float32)
        return {"pixel_values_videos": clip.unsqueeze(0)}


class _VjEncoderStub(nn.Module):
    """[N, F, C, H, W] -> [N, latent_frames * TOKENS_PER_FRAME, VJ_HIDDEN], content-dependent."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tubelet_size=2, hidden_size=VJ_HIDDEN, image_size=8)
        self.register_buffer("probe", torch.linspace(0.5, 1.5, VJ_HIDDEN))

    @property
    def device(self):
        return self.probe.device

    def get_vision_features(self, pixel_values_videos):
        clips = pixel_values_videos
        count, frames = clips.shape[0], clips.shape[1]
        latent_frames = frames // self.config.tubelet_size
        frame_means = clips.reshape(count, latent_frames, -1).mean(dim=-1)
        tokens = frame_means.repeat_interleave(TOKENS_PER_FRAME, dim=1)
        offsets = torch.arange(tokens.shape[1], dtype=torch.float32)
        return tokens[:, :, None] * self.probe + offsets[None, :, None] * 0.01


class _RetroPredictorStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_proj = nn.Linear(TEACHER, TEACHER)
        self.cond_proj = nn.Linear(TEACHER, TEACHER)
        self.last_causal = None

    def forward(self, x, actions, causal=True):
        self.last_causal = causal
        return self.state_proj(x) + self.cond_proj(actions).mean(dim=1, keepdim=True)


def _config():
    return OmegaConf.create(
        {
            "datasets": {"vla_data": {"CoT_prompt": ""}, "video_data": {"CoT_prompt": ""}},
            "trainer": {
                "retro_loss_weight": 1.0,
                "pick_loss_weight": 0.2,
                "rec_loss_fp32": True,
            },
        }
    )


def _build_model():
    torch.manual_seed(11)
    model = VLA_JEPA.__new__(VLA_JEPA)
    nn.Module.__init__(model)
    model.config = _config()
    model.capture_jepa = False
    model.last_jepa_tensors = None
    model.last_memory_diagnostics = None
    model.last_qwen_cache = None
    model.memory_enabled = True
    model.memory_schema_version = 3
    model.qwen_vl_interface = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=QWEN_HIDDEN))
    )
    model.vj_encoder = _VjEncoderStub()
    model.vj_processor = _VjProcessorStub()
    model._build_memory_modules(
        {"short_term": {"num_slots": 4, "dim": 8, "num_heads": 2}, "action_conditioning": {}}
    )
    model.vj_predictor = _RetroPredictorStub()
    return model


def _video(seed, frames=16):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(2, frames, 8, 8, 3)).astype(np.uint8)


def _examples(batch=2):
    return [{"video": _video(7 + row), "lang": "clip", "image": []} for row in range(batch)]


class RetroRunSamplerTest(unittest.TestCase):
    def test_runs_are_contiguous_interior_and_bounded(self):
        model = _build_model()
        for _ in range(200):
            start, run_len = model._sample_retro_run(8)
            self.assertGreaterEqual(run_len, 3)
            self.assertLessEqual(run_len, 6)
            self.assertGreaterEqual(start, 1)
            self.assertLessEqual(start + run_len, 8)

    def test_too_few_frames_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "latent frames"):
            _build_model()._sample_retro_run(4)


class RetroForwardTest(unittest.TestCase):
    def test_losses_diagnostics_and_bidirectional_predictor(self):
        model = _build_model()
        losses = model._forward_retro_video(_examples())
        self.assertEqual(set(losses), {"retro_loss", "pick_loss"})
        for value in losses.values():
            self.assertTrue(bool(torch.isfinite(value)))
        self.assertIs(model.vj_predictor.last_causal, False)
        diagnostics = model.last_memory_diagnostics
        self.assertIn("retro_loss_raw", diagnostics)
        self.assertGreaterEqual(float(diagnostics["pick_acc"]), 0.0)
        self.assertLessEqual(float(diagnostics["pick_acc"]), 1.0)
        self.assertNotIn("prior_gap", diagnostics)

    def test_gradients_reach_the_writer(self):
        model = _build_model()
        losses = model._forward_retro_video(_examples())
        (losses["retro_loss"] + losses["pick_loss"]).backward()
        writer_grads = [
            parameter.grad
            for parameter in model.memory_module.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(writer_grads)
        self.assertTrue(any(float(grad.abs().sum()) > 0 for grad in writer_grads))

    def test_capture_adds_prior_gap(self):
        model = _build_model()
        model.capture_jepa = True
        model._forward_retro_video(_examples())
        self.assertIn("prior_gap", model.last_memory_diagnostics)
        self.assertTrue(bool(torch.isfinite(model.last_memory_diagnostics["prior_gap"])))

    def test_forward_dispatches_video_batches_to_retro(self):
        model = _build_model()
        losses = model(_examples())
        self.assertIn("retro_loss", losses)


class PoolFrameTokensTest(unittest.TestCase):
    def test_group_mean_shape_and_value(self):
        tokens = torch.arange(2 * 8 * 4, dtype=torch.float32).view(2, 8, 4)
        pooled = VLA_JEPA._pool_frame_tokens(tokens, groups=4)
        self.assertEqual(tuple(pooled.shape), (2, 4, 4))
        torch.testing.assert_close(pooled[0, 0], tokens[0, :2].mean(dim=0))

    def test_indivisible_tokens_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "groups"):
            VLA_JEPA._pool_frame_tokens(torch.zeros(1, 6, 4), groups=8)


class RealPredictorBidirectionalTest(unittest.TestCase):
    def test_causal_false_supports_longer_t_than_the_causal_mask(self):
        torch.manual_seed(5)
        predictor = VisionTransformerPredictorAC(
            num_frames=4,
            img_size=(32, 32),
            tubelet_size=1,
            patch_size=16,
            depth=2,
            num_heads=4,
            embed_dim=TEACHER,
            action_embed_dim=TEACHER,
            num_add_tokens=2,
        )
        frames, grid = 8, 4  # T=8 exceeds the causal mask built for T=4
        x = torch.randn(1, frames * grid, TEACHER)
        cond = torch.randn(1, frames * 2, TEACHER)
        out = predictor(x, cond, causal=False)
        self.assertEqual(tuple(out.shape), (1, frames * grid, TEACHER))


if __name__ == "__main__":
    unittest.main()
