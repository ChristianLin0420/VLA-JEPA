import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from starVLA.model.framework.VLA_JEPA import (
    MemoryConditioningAdapter,
    QwenTokenBundle,
    VLA_JEPA,
    _scale_grad,
)
from starVLA.model.modules.memory import (
    RecurrentMemory,
    ResidualMemoryFusion,
    SparseKeyMemoryFusion,
)

HIDDEN = 16
MEM_DIM = 8
KEY_DIM = 4
SLOTS = 4
HEADS = 2
EMBODIED_TOKENS = 6
ACTION_TOKENS = 4
COND_TOKENS = 3
VJ_HIDDEN = 6
TEACHER_DIM = 2 * VJ_HIDDEN
NCE_DIM = 8


class _QwenEncoderStub:
    """Deterministic image-dependent tokens; records the images it received."""

    def __init__(self):
        self.received_images = None

    def __call__(self, images, instructions, prompt, require_embodied):
        self.received_images = images
        means = torch.tensor(
            [
                float(np.mean([np.asarray(view, dtype=np.float32) for view in views])) / 255.0
                for views in images
            ],
            dtype=torch.float32,
        )
        grid = torch.linspace(0.1, 1.0, HIDDEN)
        action = means[:, None, None] + grid * torch.linspace(1.0, 2.0, ACTION_TOKENS)[None, :, None]
        embodied = None
        if require_embodied:
            embodied = (
                means[:, None, None]
                + grid * torch.linspace(-2.0, -1.0, EMBODIED_TOKENS)[None, :, None]
            )
        last_hidden = torch.zeros(len(images), 3, HIDDEN)
        return QwenTokenBundle(last_hidden, action, embodied)


class _VjProcessorStub:
    def __call__(self, videos, return_tensors):
        clip = torch.as_tensor(np.asarray(videos), dtype=torch.float32)
        return {"pixel_values_videos": clip.unsqueeze(0)}


class _VjEncoderStub(nn.Module):
    """Maps clip content to latents deterministically: [N, F, C, H, W] -> [N, 4, VJ_HIDDEN]."""

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
        tokens = frame_means.repeat_interleave(2, dim=1)
        offsets = torch.arange(tokens.shape[1], dtype=torch.float32)
        return tokens[:, :, None] * self.probe + offsets[None, :, None] * 0.01


class _VjPredictorStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_proj = nn.Linear(TEACHER_DIM, TEACHER_DIM)
        self.cond_proj = nn.Linear(HIDDEN, TEACHER_DIM)

    def forward(self, input_states, actions):
        return self.state_proj(input_states) + self.cond_proj(actions).mean(dim=1, keepdim=True)


class _ActionModelStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_embodied_shape = None

    def forward(self, embodied, actions, state):
        self.last_embodied_shape = tuple(embodied.shape)
        return embodied.square().mean()

    def predict_action(self, embodied, state, generator=None, initial_noise=None):
        self.last_embodied_shape = tuple(embodied.shape)
        return embodied.new_zeros(embodied.shape[0], 2, 7) + embodied.mean()


def _config(mask_grad_alpha=0.1):
    return OmegaConf.create(
        {
            "datasets": {
                "vla_data": {"CoT_prompt": ""},
                "video_data": {"CoT_prompt": ""},
            },
            "trainer": {
                "repeated_diffusion_steps": 1,
                "robot_world_model_loss": True,
                "memory_bptt_steps": 4,
                "memory_detach_burn_in": True,
                "mask_grad_alpha": mask_grad_alpha,
            },
        }
    )


def _bare_model(config):
    model = VLA_JEPA.__new__(VLA_JEPA)
    nn.Module.__init__(model)
    model.config = config
    model.capture_jepa = False
    model.last_jepa_tensors = None
    model.jepa_num_views = 2
    model.last_memory_diagnostics = None
    model.last_qwen_cache = None
    return model


def _build_model(schema_version=2, mask_grad_alpha=0.1, rec_condition_source="policy_tokens"):
    torch.manual_seed(7)
    model = _bare_model(_config(mask_grad_alpha))
    model._rec_condition_source = rec_condition_source
    model.future_action_window_size = 1
    model.past_action_window_size = 0
    model.chunk_len = 2
    model.expected_action_token_count = COND_TOKENS
    model.expected_embodied_token_count = EMBODIED_TOKENS
    model.memory_enabled = True
    model.memory_schema_version = schema_version
    model.vj_encoder = _VjEncoderStub()
    model.vj_processor = _VjProcessorStub()
    model.vj_predictor = _VjPredictorStub()
    model.action_model = _ActionModelStub()
    model._encode_qwen_tokens = _QwenEncoderStub()
    if schema_version >= 2:
        model.memory_module = RecurrentMemory(
            source_dim=HIDDEN,
            num_slots=SLOTS,
            memory_dim=MEM_DIM,
            num_heads=HEADS,
            key_dim=KEY_DIM,
            use_keys=True,
        )
        model.policy_memory_fusion = SparseKeyMemoryFusion(
            consumer_dim=HIDDEN, memory_dim=MEM_DIM, key_dim=KEY_DIM, num_slots=SLOTS
        )
        model.wm_mask_token = nn.Parameter(torch.randn(TEACHER_DIM) * 0.02)
        adapter_inputs = (
            EMBODIED_TOKENS + 1
            if rec_condition_source == "policy_tokens"
            else ACTION_TOKENS
        )
        model.mem_cond_adapter = MemoryConditioningAdapter(
            adapter_inputs, COND_TOKENS, HIDDEN, rank=4
        )
        model.nce_head_h = nn.Sequential(
            nn.Linear(HIDDEN, NCE_DIM), nn.GELU(), nn.Linear(NCE_DIM, NCE_DIM)
        )
        model.nce_head_g = nn.Sequential(
            nn.Linear(TEACHER_DIM, NCE_DIM), nn.GELU(), nn.Linear(NCE_DIM, NCE_DIM)
        )
    else:
        model.memory_module = RecurrentMemory(
            source_dim=HIDDEN, num_slots=SLOTS, memory_dim=MEM_DIM, num_heads=HEADS
        )
        model.policy_memory_fusion = ResidualMemoryFusion(
            consumer_dim=HIDDEN, memory_dim=MEM_DIM, bottleneck_dim=MEM_DIM, num_heads=HEADS
        )
    return model


def _robot_step(seed, with_clean=False):
    rng = np.random.default_rng(seed)
    video = rng.integers(60, 200, size=(2, 4, 8, 8, 3)).astype(np.uint8)
    step = {
        "action": rng.uniform(-1.0, 1.0, size=(2, 7)).astype(np.float32),
        "image": [
            Image.fromarray(rng.integers(30, 220, size=(8, 8, 3)).astype(np.uint8))
            for _ in range(2)
        ],
        "lang": "stub task",
        "video": video,
    }
    if with_clean:
        step["video_clean"] = video.copy()
    return step


def _segment(mask_plan):
    return {
        "steps": [_robot_step(1), _robot_step(2), _robot_step(3, with_clean=True)],
        "sequence_valid": np.array([True, True, True]),
        "loss_mask": np.array([False, True, True]),
        "update_mask": np.array([True, True, True]),
        "is_first": np.array([False, False, False]),
        "mask_plan": mask_plan,
    }


class ScaleGradTest(unittest.TestCase):
    def test_identity_forward_scaled_backward(self):
        tokens = torch.randn(2, 3, requires_grad=True)
        weights = torch.randn(2, 3)
        scaled = _scale_grad(tokens, 0.25)
        self.assertTrue(torch.equal(scaled, tokens))
        (scaled * weights).sum().backward()
        torch.testing.assert_close(tokens.grad, weights * 0.25)


class MemoryConditioningAdapterTest(unittest.TestCase):
    def test_channel_map_is_identity_at_init(self):
        torch.manual_seed(3)
        adapter = MemoryConditioningAdapter(7, 3, HIDDEN, rank=4)
        tokens = torch.randn(2, 7, HIDDEN)
        mixed = adapter.token_mixer(tokens.transpose(1, 2)).transpose(1, 2)
        self.assertEqual(tuple(mixed.shape), (2, 3, HIDDEN))
        torch.testing.assert_close(adapter(tokens), mixed)
        torch.testing.assert_close(
            adapter.channel_up(adapter.channel_down(mixed)), torch.zeros_like(mixed)
        )


class ReconLossTest(unittest.TestCase):
    def test_identity_at_init_matches_mixer_only_decode(self):
        model = _build_model()
        clip = _robot_step(21, with_clean=True)["video_clean"]
        policy = torch.randn(1, EMBODIED_TOKENS + 1, HIDDEN)
        markers = torch.randn(1, ACTION_TOKENS, HIDDEN)
        loss = model._compute_recon_loss([clip], policy, markers)

        latents, tokens_per_frame, latent_frames = model._encode_video_latents([clip], policy)
        gt_states = latents[:, tokens_per_frame:, :]
        input_states = model.wm_mask_token[None, None, :].expand(
            gt_states.shape[0], tokens_per_frame * (latent_frames - 1), -1
        )
        mixer_only = model.mem_cond_adapter.token_mixer(policy.transpose(1, 2)).transpose(1, 2)
        manual = F.l1_loss(model.vj_predictor(input_states, mixer_only), gt_states)
        torch.testing.assert_close(loss, manual)

    def test_policy_gradient_scaled_by_alpha(self):
        clip = _robot_step(22, with_clean=True)["video_clean"]
        policy = torch.randn(1, EMBODIED_TOKENS + 1, HIDDEN)

        grads = {}
        for alpha in (0.1, 1.0):
            model = _build_model(mask_grad_alpha=alpha)
            leaf = policy.clone().requires_grad_(True)
            markers = torch.randn(1, ACTION_TOKENS, HIDDEN)
            model._compute_recon_loss([clip], leaf, markers).backward()
            grads[alpha] = leaf.grad
        self.assertGreater(float(grads[1.0].abs().sum()), 0.0)
        torch.testing.assert_close(grads[0.1], grads[1.0] * 0.1)

    def test_private_decoder_arm_detaches_policy_tokens(self):
        model = _build_model(rec_condition_source="detached_action_tokens")
        clip = _robot_step(23, with_clean=True)["video_clean"]
        policy = torch.randn(1, EMBODIED_TOKENS + 1, HIDDEN, requires_grad=True)
        markers = torch.randn(1, ACTION_TOKENS, HIDDEN, requires_grad=True)
        loss = model._compute_recon_loss([clip], policy, markers)
        loss.backward()
        self.assertIsNone(policy.grad)
        self.assertIsNone(markers.grad)
        self.assertEqual(model.mem_cond_adapter.token_mixer.in_features, ACTION_TOKENS)

    def test_fp32_branch_upcasts_low_precision_conditioning(self):
        # Without the explicit casts, bf16 activations meet fp32 adapter
        # weights and the matmul raises; a finite fp32 loss proves the
        # branch owns its precision end to end (memv2.4, diagnostic D2).
        model = _build_model()
        model.config.trainer["rec_loss_fp32"] = True
        clip = _robot_step(24, with_clean=True)["video_clean"]
        policy = torch.randn(1, EMBODIED_TOKENS + 1, HIDDEN).to(torch.bfloat16)
        markers = torch.randn(1, ACTION_TOKENS, HIDDEN).to(torch.bfloat16)
        loss = model._compute_recon_loss([clip], policy, markers)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_fp32_flag_off_preserves_existing_semantics(self):
        clip = _robot_step(25, with_clean=True)["video_clean"]
        policy = torch.randn(1, EMBODIED_TOKENS + 1, HIDDEN)
        markers = torch.randn(1, ACTION_TOKENS, HIDDEN)
        baseline = _build_model()._compute_recon_loss([clip], policy, markers)
        flagged = _build_model()
        flagged.config.trainer["rec_loss_fp32"] = True
        torch.testing.assert_close(
            flagged._compute_recon_loss([clip], policy, markers), baseline
        )


class ForwardOneMaskTest(unittest.TestCase):
    def test_masked_step_switches_losses_blacks_inputs_and_gates_write(self):
        model = _build_model()
        state = model.memory_module.init_state(1, device=torch.device("cpu"))
        step = _robot_step(5, with_clean=True)

        losses, state_after, _ = model._forward_one([step], memory_state=state, masked=True)
        self.assertIn("action_loss", losses)
        self.assertIn("rec_loss", losses)
        self.assertNotIn("wm_loss", losses)

        # Blacking happened before the encoder stub, without mutating the sample.
        for view in model._encode_qwen_tokens.received_images[0]:
            self.assertEqual(int(np.asarray(view).max()), 0)
        self.assertGreater(int(np.asarray(step["image"][0]).max()), 0)
        self.assertGreater(int(step["video"].max()), 0)

        # Phase A: no black-frame write, but the decision clock ticks.
        torch.testing.assert_close(state_after.working, state.working)
        torch.testing.assert_close(state_after.keys, state.keys)
        self.assertEqual(int(state_after.steps), 1)

    def test_unmasked_step_keeps_world_loss_and_writes(self):
        model = _build_model()
        state = model.memory_module.init_state(1, device=torch.device("cpu"))
        step = _robot_step(6)

        losses, state_after, _ = model._forward_one([step], memory_state=state, masked=False)
        self.assertIn("wm_loss", losses)
        self.assertNotIn("rec_loss", losses)
        self.assertFalse(torch.equal(state_after.working, state.working))
        self.assertFalse(torch.equal(state_after.keys, state.keys))
        self.assertEqual(int(state_after.steps), 1)

    def test_masked_step_requires_schema_two(self):
        model = _build_model(schema_version=1)
        with self.assertRaises(ValueError):
            model._forward_one([_robot_step(7, with_clean=True)], masked=True)


class ForwardSequenceTest(unittest.TestCase):
    def test_mask_plan_produces_rec_wm_and_nce_outputs(self):
        model = _build_model()
        output = model.forward_sequence([_segment(mask_plan=[False, True])])
        self.assertEqual(
            set(output), {"action_loss", "wm_loss", "rec_loss", "nce_anchor", "nce_positive"}
        )
        self.assertEqual(tuple(output["nce_anchor"].shape), (2, NCE_DIM))
        self.assertEqual(tuple(output["nce_positive"].shape), (2, NCE_DIM))
        self.assertTrue(output["nce_anchor"].requires_grad)
        self.assertTrue(output["nce_positive"].requires_grad)
        torch.testing.assert_close(
            output["nce_anchor"].norm(dim=-1), torch.ones(2)
        )
        torch.testing.assert_close(
            output["nce_positive"].norm(dim=-1), torch.ones(2)
        )
        # One shared per-segment positive, repeated per supervised anchor.
        torch.testing.assert_close(output["nce_positive"][0], output["nce_positive"][1])

    def test_unmasked_schema_two_segment_has_no_rec_loss(self):
        model = _build_model()
        output = model.forward_sequence([_segment(mask_plan=[False, False])])
        self.assertEqual(
            set(output), {"action_loss", "wm_loss", "nce_anchor", "nce_positive"}
        )

    def test_schema_one_rejects_active_mask_plan(self):
        model = _build_model(schema_version=1)
        with self.assertRaises(ValueError):
            model.forward_sequence([_segment(mask_plan=[False, True])])

    def test_schema_one_output_is_memv1(self):
        model = _build_model(schema_version=1)
        output = model.forward_sequence([_segment(mask_plan=None)])
        self.assertEqual(set(output), {"action_loss", "wm_loss"})


class ConstructionTest(unittest.TestCase):
    def _memory_cfg(self, schema_version):
        return OmegaConf.create(
            {
                "enabled": True,
                "schema_version": schema_version,
                "short_term": {
                    "enabled": True,
                    "num_slots": SLOTS,
                    "dim": MEM_DIM,
                    "num_heads": HEADS,
                    "update_gate_init": 0.1,
                    "key_dim": KEY_DIM,
                },
                "action_conditioning": {
                    "enabled": True,
                    "bottleneck_dim": MEM_DIM,
                    "dropout": 0.0,
                    "zero_init_gate": False,
                    "gate_init": 1.0e-3,
                },
            }
        )

    def _constructed(self, schema_version, memory_cfg=None):
        model = _bare_model(_config())
        model.qwen_vl_interface = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(hidden_size=HIDDEN))
        )
        model.vj_encoder = _VjEncoderStub()
        model.expected_action_token_count = COND_TOKENS
        model.expected_embodied_token_count = EMBODIED_TOKENS
        model.memory_enabled = True
        model.memory_schema_version = schema_version
        model.memory_module = None
        model.policy_memory_fusion = None
        model._build_memory_modules(memory_cfg or self._memory_cfg(schema_version))
        return model

    def test_schema_one_builds_bit_identical_memv1_module_tree(self):
        torch.manual_seed(11)
        model = self._constructed(schema_version=1)
        torch.manual_seed(11)
        reference_memory = RecurrentMemory(
            source_dim=HIDDEN,
            num_slots=SLOTS,
            memory_dim=MEM_DIM,
            num_heads=HEADS,
            update_gate_init=0.1,
        )
        reference_fusion = ResidualMemoryFusion(
            consumer_dim=HIDDEN,
            memory_dim=MEM_DIM,
            bottleneck_dim=MEM_DIM,
            num_heads=HEADS,
            dropout=0.0,
            gate_init=1.0e-3,
        )
        for built, reference in (
            (model.memory_module, reference_memory),
            (model.policy_memory_fusion, reference_fusion),
        ):
            built_state = built.state_dict()
            reference_state = reference.state_dict()
            self.assertEqual(set(built_state), set(reference_state))
            for key, value in reference_state.items():
                self.assertTrue(torch.equal(built_state[key], value), key)
        self.assertIsInstance(model.policy_memory_fusion, ResidualMemoryFusion)
        for name in ("wm_mask_token", "mem_cond_adapter", "nce_head_h", "nce_head_g"):
            self.assertFalse(hasattr(model, name), name)

    def test_schema_two_builds_memv2_modules(self):
        model = self._constructed(schema_version=2)
        self.assertIsInstance(model.policy_memory_fusion, SparseKeyMemoryFusion)
        self.assertTrue(model.memory_module.use_keys)
        self.assertEqual(tuple(model.wm_mask_token.shape), (TEACHER_DIM,))
        self.assertEqual(
            tuple(model.mem_cond_adapter.token_mixer.weight.shape),
            (COND_TOKENS, EMBODIED_TOKENS + 1),
        )
        self.assertEqual(model.nce_head_h[-1].out_features, 256)
        self.assertEqual(model.nce_head_g[0].in_features, TEACHER_DIM)
        # Absent config key defaults to the closed gate.
        self.assertEqual(float(model.policy_memory_fusion.gamma_c), 0.0)

    def test_schema_two_threads_content_gate_init(self):
        cfg = self._memory_cfg(schema_version=2)
        cfg.action_conditioning.content_gate_init = 0.05
        model = self._constructed(schema_version=2, memory_cfg=cfg)
        self.assertEqual(
            float(model.policy_memory_fusion.gamma_c), float(torch.tensor(0.05))
        )


class PredictActionTest(unittest.TestCase):
    def test_schema_two_feeds_33_style_tokens_and_roundtrips_keys(self):
        model = _build_model()
        model.eval()
        images = [_robot_step(9)["image"]]

        output, state = model.predict_action(
            batch_images=images, instructions=["stub"], return_memory_state=True
        )
        self.assertEqual(
            model.action_model.last_embodied_shape, (1, EMBODIED_TOKENS + 1, HIDDEN)
        )
        self.assertEqual(output["normalized_actions"].shape, (1, 2, 7))
        self.assertIsNotNone(state.keys)
        self.assertEqual(tuple(state.keys.shape), (1, SLOTS, KEY_DIM))
        self.assertEqual(int(state.steps), 1)

        _, state_next = model.predict_action(
            batch_images=images,
            instructions=["stub"],
            memory_state=state,
            return_memory_state=True,
        )
        self.assertEqual(int(state_next.steps), 2)
        self.assertFalse(torch.equal(state_next.keys, state.keys))


if __name__ == "__main__":
    unittest.main()
