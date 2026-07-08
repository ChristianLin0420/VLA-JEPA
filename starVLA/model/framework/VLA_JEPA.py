# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.memory import (
    MemoryState,
    RecurrentMemory,
    ResidualMemoryFusion,
    SparseKeyMemoryFusion,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenTokenBundle:
    """Validated Qwen tensors shared by training and inference paths."""

    last_hidden: torch.Tensor
    action_tokens: torch.Tensor
    embodied_action_tokens: Optional[torch.Tensor]


class _ScaledGradient(torch.autograd.Function):
    """Identity in the forward pass; scales the gradient by ``alpha``."""

    @staticmethod
    def forward(ctx, tokens: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return tokens.view_as(tokens)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.alpha, None


def _scale_grad(tokens: torch.Tensor, alpha: float) -> torch.Tensor:
    return _ScaledGradient.apply(tokens, alpha)


class MemoryConditioningAdapter(nn.Module):
    """Map fused policy tokens onto predictor-compatible conditioning tokens.

    A learned linear mixer over the token dimension followed by a low-rank
    residual channel map whose up-projection is zero-initialized, so the
    adapter is exactly the token mixer at initialization.
    """

    def __init__(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        dim: int,
        rank: int = 256,
    ) -> None:
        super().__init__()
        self.token_mixer = nn.Linear(num_input_tokens, num_output_tokens)
        self.channel_down = nn.Linear(dim, rank)
        self.channel_up = nn.Linear(rank, dim)
        nn.init.zeros_(self.channel_up.weight)
        nn.init.zeros_(self.channel_up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mixer(tokens.transpose(1, 2)).transpose(1, 2)
        return mixed + self.channel_up(self.channel_down(mixed))

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # --- JEPA representation-analysis side-channel -------------------------------
        # The trainer flips `capture_jepa` on logging steps; forward() then stashes the
        # (detached) predictor tensors here so the trainer can compute representation
        # metrics WITHOUT polluting the loss dict (which the trainer sums over). Off
        # logging steps this is a no-op (zero overhead). `num_views` lets the analysis
        # split the multi-view-concatenated latent dim per camera.
        self.capture_jepa = False
        self.last_jepa_tensors = None
        self.jepa_num_views = 2

        self.expected_action_token_count = (
            self.config.framework.vj2_model.num_frames // tubelet_size - 1
        ) * self.config.framework.vj2_model.num_action_tokens_per_timestep
        self.expected_embodied_token_count = int(
            self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction
        )

        # Missing memory configuration remains exactly checkpoint-compatible with
        # the legacy model: no learned memory module is constructed unless enabled.
        memory_cfg = self.config.framework.get("memory", {})
        self.memory_enabled = bool(memory_cfg.get("enabled", False))
        self.memory_schema_version = int(memory_cfg.get("schema_version", 0)) if self.memory_enabled else 0
        self.memory_module = None
        self.policy_memory_fusion = None
        self.last_memory_diagnostics = None
        self.last_qwen_cache = None
        self._rec_condition_source = "policy_tokens"
        if self.memory_enabled:
            self._build_memory_modules(memory_cfg)

    def _build_memory_modules(self, memory_cfg) -> None:
        short_cfg = memory_cfg.get("short_term", {})
        action_cfg = memory_cfg.get("action_conditioning", {})
        if not bool(short_cfg.get("enabled", True)):
            raise ValueError("Phase-1 memory requires framework.memory.short_term.enabled=true")
        if not bool(action_cfg.get("enabled", True)):
            raise ValueError("Phase-1 memory requires framework.memory.action_conditioning.enabled=true")
        if bool(memory_cfg.get("long_term", {}).get("enabled", False)):
            raise NotImplementedError("long-term associative memory is a Phase-2 feature")
        if bool(memory_cfg.get("world_model_conditioning", {}).get("enabled", False)):
            raise NotImplementedError("world-model memory conditioning is a Phase-3 feature")

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        memory_dim = int(short_cfg.get("dim", 512))
        if self.memory_schema_version == 3:
            # memv3 Retro-JEPA: the writer consumes pooled frozen V-JEPA2
            # latents (so Stage M1 runs on pure video), the retrodictor is the
            # shared vj_predictor, and the policy read is native attention —
            # no fusion module exists for the optimizer to squelch.
            predictor_dim = int(self.vj_encoder.config.hidden_size) * 2
            self.memory_module = RecurrentMemory(
                source_dim=predictor_dim,
                num_slots=int(short_cfg.get("num_slots", 8)),
                memory_dim=memory_dim,
                num_heads=int(short_cfg.get("num_heads", 8)),
                update_gate_init=float(short_cfg.get("update_gate_init", 0.1)),
            )
            self.wm_mask_token = nn.Parameter(torch.empty(predictor_dim))
            nn.init.normal_(self.wm_mask_token, mean=0.0, std=0.02)
            self.retro_cond_proj = nn.Linear(memory_dim, predictor_dim)
            self.retro_pick_head = nn.Sequential(
                nn.Linear(predictor_dim, 256), nn.GELU(), nn.Linear(256, 256)
            )
            # Stage M2 reader: read tokens appended to the action expert's
            # cross-attention context (unused in Stage M1).
            self.memory_read_proj = nn.Linear(memory_dim, hidden_dim)
            return
        if self.memory_schema_version >= 2:
            num_slots = int(short_cfg.get("num_slots", 8))
            key_dim = int(short_cfg.get("key_dim", 128))
            self.memory_module = RecurrentMemory(
                source_dim=hidden_dim,
                num_slots=num_slots,
                memory_dim=memory_dim,
                num_heads=int(short_cfg.get("num_heads", 8)),
                update_gate_init=float(short_cfg.get("update_gate_init", 0.1)),
                key_dim=key_dim,
                use_keys=True,
            )
            self.policy_memory_fusion = SparseKeyMemoryFusion(
                consumer_dim=hidden_dim,
                memory_dim=memory_dim,
                key_dim=key_dim,
                num_slots=num_slots,
                content_gate_init=float(action_cfg.get("content_gate_init", 0.0)),
                content_gate_fixed=bool(action_cfg.get("content_gate_fixed", False)),
            )
            # D2 diagnosis: at the default amplitude the content contrast is below
            # bf16 training arithmetic; content_scale lifts the fixed injection.
            self.policy_memory_fusion.residual_scale = float(
                action_cfg.get("content_scale", 1.0)
            )
            predictor_dim = int(self.vj_encoder.config.hidden_size) * 2
            self.wm_mask_token = nn.Parameter(torch.empty(predictor_dim))
            nn.init.normal_(self.wm_mask_token, mean=0.0, std=0.02)
            # Pre-registered consumer-tie A/B: the private-decoder arm conditions
            # reconstruction on detached action markers instead of policy tokens.
            self._rec_condition_source = str(
                memory_cfg.get("rec_condition_source", "policy_tokens")
            )
            if self._rec_condition_source not in ("policy_tokens", "detached_action_tokens"):
                raise ValueError(
                    f"unsupported rec_condition_source: {self._rec_condition_source}"
                )
            adapter_input_tokens = (
                self.expected_embodied_token_count + 1
                if self._rec_condition_source == "policy_tokens"
                else self.expected_action_token_count
            )
            self.mem_cond_adapter = MemoryConditioningAdapter(
                num_input_tokens=adapter_input_tokens,
                num_output_tokens=self.expected_action_token_count,
                dim=hidden_dim,
            )
            self.nce_head_h = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, 256)
            )
            self.nce_head_g = nn.Sequential(
                nn.Linear(predictor_dim, 256), nn.GELU(), nn.Linear(256, 256)
            )
        else:
            self.memory_module = RecurrentMemory(
                source_dim=hidden_dim,
                num_slots=int(short_cfg.get("num_slots", 8)),
                memory_dim=memory_dim,
                num_heads=int(short_cfg.get("num_heads", 8)),
                update_gate_init=float(short_cfg.get("update_gate_init", 0.1)),
            )
            zero_gate = bool(action_cfg.get("zero_init_gate", True))
            gate_init = 0.0 if zero_gate else float(action_cfg.get("gate_init", 1.0e-3))
            self.policy_memory_fusion = ResidualMemoryFusion(
                consumer_dim=hidden_dim,
                memory_dim=memory_dim,
                bottleneck_dim=int(action_cfg.get("bottleneck_dim", memory_dim)),
                num_heads=int(action_cfg.get("num_heads", short_cfg.get("num_heads", 8))),
                dropout=float(action_cfg.get("dropout", 0.0)),
                gate_init=gate_init,
            )

    def _maybe_capture_jepa(self, predicted_states, gt_states, input_states, action_tokens):
        """Stash detached predictor tensors for representation analysis (logging steps only).

        When capture is off we MUST NOT clear last_jepa_tensors: the cotrain runs two
        forwards per step (VLA then video) and toggles capture off before the video
        forward, so clobbering here would wipe the VLA-pass capture before the trainer
        reads it. The trainer clears last_jepa_tensors after logging instead.
        """
        if not getattr(self, "capture_jepa", False):
            return
        try:
            self.last_jepa_tensors = {
                "predicted": predicted_states.detach(),
                "gt": gt_states.detach(),
                "input": input_states.detach(),
                "action_tokens": action_tokens.detach() if action_tokens is not None else None,
            }
        except Exception:
            self.last_jepa_tensors = None

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    @staticmethod
    def _autocast_context(tensor: torch.Tensor, dtype: torch.dtype):
        if tensor.is_cuda and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    @staticmethod
    def _select_token_rows(
        last_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_ids: List[int],
        expected_count: int,
        label: str,
    ) -> torch.Tensor:
        token_id_tensor = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
        mask = torch.isin(input_ids, token_id_tensor)
        counts = mask.sum(dim=1)
        if not torch.all(counts == expected_count):
            raise ValueError(
                f"expected exactly {expected_count} {label} tokens per row; got {counts.tolist()}"
            )
        batch_size, _, hidden_dim = last_hidden.shape
        return last_hidden[mask].reshape(batch_size, expected_count, hidden_dim)

    def _encode_qwen_tokens(
        self,
        images: List[List[Image.Image]],
        instructions: List[str],
        prompt_template: str,
        require_embodied: bool,
    ) -> QwenTokenBundle:
        replacements = {"{actions}": self.replace_prompt}
        if require_embodied:
            replacements["{e_actions}"] = self.embodied_replace_prompt
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            prompt_replace_dict=replacements,
            prompt_template=prompt_template,
        )
        with self._autocast_context(qwen_inputs["input_ids"], torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            action_tokens = self._select_token_rows(
                last_hidden,
                qwen_inputs["input_ids"],
                self.action_token_ids,
                self.expected_action_token_count,
                "action-marker",
            )
            embodied = None
            if require_embodied:
                embodied = self._select_token_rows(
                    last_hidden,
                    qwen_inputs["input_ids"],
                    [self.embodied_action_token_id],
                    self.expected_embodied_token_count,
                    "embodied-action",
                )
        return QwenTokenBundle(last_hidden, action_tokens, embodied)

    def _encode_video_latents(
        self,
        batch_videos: List[np.ndarray],
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """Frozen-teacher V-JEPA latents for a batch of multi-view clips."""
        videos = np.stack(batch_videos).transpose(0, 1, 2, 5, 3, 4)
        batch_size, num_views, num_frames, channels, height, width = videos.shape
        videos = videos.reshape(batch_size * num_views, num_frames, channels, height, width)
        processed = [
            self.vj_processor(videos=videos[i], return_tensors="pt")["pixel_values_videos"].to(
                self.vj_encoder.device
            )
            for i in range(batch_size * num_views)
        ]
        input_videos = torch.cat(processed, dim=0)
        with self._autocast_context(reference, torch.bfloat16):
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=num_views, dim=0), dim=2)
        latent_frames = num_frames // self.vj_encoder.config.tubelet_size
        tokens_per_frame = video_embeddings.shape[1] // latent_frames
        return video_embeddings, tokens_per_frame, latent_frames

    def _compute_world_loss(
        self,
        batch_videos: List[np.ndarray],
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        video_embeddings, tokens_per_frame, latent_frames = self._encode_video_latents(
            batch_videos, action_tokens
        )
        with self._autocast_context(action_tokens, torch.bfloat16):
            input_states = video_embeddings[:, : tokens_per_frame * (latent_frames - 1), :]
            gt_states = video_embeddings[:, tokens_per_frame:, :]
            predicted_states = self.vj_predictor(input_states, action_tokens)
            world_loss = F.l1_loss(predicted_states, gt_states, reduction="mean")
        self._maybe_capture_jepa(predicted_states, gt_states, input_states, action_tokens)
        return world_loss

    def _compute_recon_loss(
        self,
        batch_videos_clean: List[np.ndarray],
        policy_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Masked latent reconstruction decoded through the policy's conditioning tensor."""
        alpha = float(self.config.trainer.get("mask_grad_alpha", 0.1))
        video_embeddings, tokens_per_frame, latent_frames = self._encode_video_latents(
            batch_videos_clean, policy_tokens
        )
        # D2 (memv2.2): under bf16 the live-vs-foreign loss contrast at unit
        # content amplitude sits below the training arithmetic; the fp32
        # branch lowers that floor ~250x so moderate amplitudes stay
        # trainable.  Weights are fp32 masters either way — only the
        # activations change.
        rec_fp32 = bool(self.config.trainer.get("rec_loss_fp32", False))
        if rec_fp32 and policy_tokens.is_cuda:
            # nullcontext would leave the trainer's outer bf16 autocast
            # active and silently re-demote the branch.
            rec_context = torch.autocast("cuda", enabled=False)
        else:
            rec_context = self._autocast_context(policy_tokens, torch.bfloat16)
        with rec_context:
            gt_states = video_embeddings[:, tokens_per_frame:, :]
            input_states = self.wm_mask_token[None, None, :].expand(
                gt_states.shape[0], tokens_per_frame * (latent_frames - 1), -1
            )
            if self._rec_condition_source == "detached_action_tokens":
                cond_input = action_tokens.detach()
            else:
                cond_input = _scale_grad(policy_tokens, alpha)
            if rec_fp32:
                gt_states = gt_states.float()
                input_states = input_states.float()
                cond_input = cond_input.float()
            conditioning = self.mem_cond_adapter(cond_input)
            predicted_states = self.vj_predictor(input_states, conditioning)
            return F.l1_loss(predicted_states, gt_states, reduction="mean")

    @staticmethod
    def _pool_frame_tokens(frame_tokens: torch.Tensor, groups: int = 8) -> torch.Tensor:
        """Group-mean [B, tokens, D] -> [B, groups, D] (parameter-free writer source)."""
        B, N, D = frame_tokens.shape
        if N % groups != 0:
            raise ValueError(f"{N} frame tokens do not divide into {groups} groups")
        return frame_tokens.view(B, groups, N // groups, D).mean(dim=2)

    def _sample_retro_run(self, latent_frames: int) -> Tuple[int, int]:
        """Contiguous masked run over past latent frames; frame 0 stays visible.

        Interior positions of a k>=3 run have both neighbours masked (the
        memory-only targets of the design doc); k in {3..6} keeps >=1 frame
        visible for context at latent_frames=8.
        """
        max_len = min(6, latent_frames - 2)
        if max_len < 3:
            raise ValueError("retrodiction needs >= 5 latent frames")
        run_len = int(torch.randint(3, max_len + 1, (1,)).item())
        start = int(torch.randint(1, latent_frames - run_len + 1, (1,)).item())
        return start, run_len

    def _retro_predict(
        self, context: torch.Tensor, read_tokens: torch.Tensor, latent_frames: int
    ) -> torch.Tensor:
        """Shared vj_predictor, bidirectional, conditioned on memory read tokens."""
        B = context.shape[0]
        cond = self.retro_cond_proj(read_tokens.to(context.dtype))
        cond = cond.unsqueeze(1).expand(B, latent_frames, -1, -1).reshape(B, -1, cond.shape[-1])
        return self.vj_predictor(context, cond, causal=False)

    def _forward_retro_video(self, examples: List[dict]) -> Dict[str, torch.Tensor]:
        """Stage M1: masked-past latent retrodiction from the recurrent memory.

        The writer rolls causally over every latent frame; a contiguous run of
        past frames is masked in the retrodictor's context, and the memory
        read is the only supply line for the interior run positions.  L_retro
        regresses frozen-encoder latents in fp32; L_pick must identify each
        masked frame among all pooled frames in the batch (same-episode and
        cross-episode negatives).  BC never appears in this stage.
        """
        reference = next(self.retro_cond_proj.parameters())
        z, tokens_per_frame, latent_frames = self._encode_video_latents(
            [example["video"] for example in examples], reference
        )
        B = z.shape[0]
        frames = z.view(B, latent_frames, tokens_per_frame, z.shape[-1])

        state = self.memory_module.init_state(B, device=reference.device)
        update_mask = torch.ones(B, dtype=torch.bool, device=reference.device)
        for index in range(latent_frames):
            state = self.memory_module.write(
                self._pool_frame_tokens(frames[:, index]), state, update_mask=update_mask
            )
        pooled_last = self._pool_frame_tokens(frames[:, -1])
        read_tokens = self.memory_module.read(pooled_last, state).tokens

        start, run_len = self._sample_retro_run(latent_frames)
        masked = torch.zeros(latent_frames, dtype=torch.bool, device=reference.device)
        masked[start : start + run_len] = True
        mask_token = self.wm_mask_token.to(frames.dtype)[None, None, None, :]
        context = torch.where(masked[None, :, None, None], mask_token, frames)
        predicted = self._retro_predict(
            context.reshape(B, latent_frames * tokens_per_frame, -1), read_tokens, latent_frames
        ).view(B, latent_frames, tokens_per_frame, -1)

        rec_fp32 = bool(self.config.trainer.get("rec_loss_fp32", True))
        with torch.autocast("cuda", enabled=False) if reference.is_cuda else nullcontext():
            pred_masked = predicted[:, masked]
            target = frames[:, masked].detach()
            if rec_fp32:
                pred_masked, target = pred_masked.float(), target.float()
            retro_loss = F.l1_loss(pred_masked, target, reduction="mean")

            # L_pick: every masked frame must identify its own frozen latent
            # among all pooled frames of the batch (a template cannot).
            anchors = F.normalize(self.retro_pick_head(pred_masked.mean(dim=2).float()), dim=-1)
            candidates = F.normalize(
                self.retro_pick_head(frames.mean(dim=2).detach().float()), dim=-1
            )
            logits = anchors.reshape(-1, anchors.shape[-1]) @ candidates.reshape(
                -1, candidates.shape[-1]
            ).T / 0.07
            masked_index = torch.nonzero(masked, as_tuple=False).squeeze(-1)
            labels = (
                torch.arange(B, device=logits.device)[:, None] * latent_frames
                + masked_index[None, :]
            ).reshape(-1)
            pick_loss = F.cross_entropy(logits, labels)
            pick_acc = (logits.argmax(dim=-1) == labels).float().mean()

        diagnostics = {
            "retro_loss_raw": retro_loss.detach(),
            "pick_acc": pick_acc.detach(),
            "masked_frames": torch.tensor(float(run_len)),
        }
        if getattr(self, "capture_jepa", False):
            with torch.no_grad():
                prior = self.memory_module.init_state(B, device=reference.device)
                prior_read = self.memory_module.read(pooled_last, prior).tokens
                prior_pred = self._retro_predict(
                    context.reshape(B, latent_frames * tokens_per_frame, -1),
                    prior_read,
                    latent_frames,
                ).view(B, latent_frames, tokens_per_frame, -1)
                prior_loss = F.l1_loss(
                    prior_pred[:, masked].float(), target, reduction="mean"
                )
            diagnostics["prior_gap"] = (prior_loss - retro_loss).detach()
        self.last_memory_diagnostics = diagnostics

        retro_weight = float(self.config.trainer.get("retro_loss_weight", 1.0))
        pick_weight = float(self.config.trainer.get("pick_loss_weight", 0.2))
        return {
            "retro_loss": retro_weight * retro_loss,
            "pick_loss": pick_weight * pick_loss,
        }

    @staticmethod
    def _black_step_inputs(examples: List[dict]) -> List[dict]:
        """Blacked shallow copies of robot samples; the originals stay intact."""
        blacked = []
        for example in examples:
            masked = dict(example)
            masked["image"] = [Image.new(image.mode, image.size) for image in example["image"]]
            masked["video"] = np.zeros_like(example["video"])
            blacked.append(masked)
        return blacked

    def _compute_action_loss(
        self,
        actions: List[np.ndarray],
        proprio_state: Optional[List[np.ndarray]],
        embodied_tokens: torch.Tensor,
    ) -> torch.Tensor:
        device = embodied_tokens.device
        dtype = embodied_tokens.dtype
        actions_tensor = torch.as_tensor(np.array(actions), device=device, dtype=dtype)
        actions_target = actions_tensor[:, -(self.future_action_window_size + 1):, :]
        # Preserve the legacy trainer-controlled repeat count for checkpoint and
        # loss-scale parity.  Memory fusion happens once before this expansion.
        repeated_steps = int(self.config.trainer.get("repeated_diffusion_steps", 4))
        actions_repeated = actions_target.repeat(repeated_steps, 1, 1)
        embodied_repeated = embodied_tokens.repeat(repeated_steps, 1, 1)
        state_repeated = None
        if proprio_state is not None:
            state_tensor = torch.as_tensor(np.array(proprio_state), device=device, dtype=dtype)
            state_repeated = state_tensor.repeat(repeated_steps, 1, 1)
        # CUDA autocast supports FP16/BF16 only.  Requesting FP32 here disables
        # the trainer's outer autocast and leaves BF16 activations facing FP32
        # Linear weights.  Keep the action head under explicit BF16 autocast.
        with self._autocast_context(embodied_tokens, torch.bfloat16):
            return self.action_model(embodied_repeated, actions_repeated, state_repeated)

    def _forward_one(
        self,
        examples: List[dict],
        memory_state: Optional[MemoryState] = None,
        reset_mask: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
        include_action_loss: bool = True,
        include_world_loss: bool = True,
        fusion_bypass: bool = False,
        masked: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[MemoryState], Dict[str, torch.Tensor]]:
        if not examples:
            raise ValueError("examples must contain at least one sample")
        has_actions = "action" in examples[0]
        if any(("action" in example) != has_actions for example in examples):
            raise ValueError("a decision batch cannot mix robot and video-only samples")
        if masked and (self.memory_schema_version < 2 or not has_actions):
            raise ValueError(
                "masked decisions require robot samples and framework.memory.schema_version >= 2"
            )

        inputs = self._black_step_inputs(examples) if masked else examples
        images = [example["image"] for example in inputs]
        instructions = [example["lang"] for example in inputs]
        prompt = (
            self.config.datasets.vla_data.get("CoT_prompt", "")
            if has_actions
            else self.config.datasets.video_data.get("CoT_prompt", "")
        )
        qwen = self._encode_qwen_tokens(images, instructions, prompt, require_embodied=has_actions)
        batch_size = qwen.last_hidden.shape[0]
        device = qwen.last_hidden.device
        active_mask = (
            torch.ones(batch_size, dtype=torch.bool, device=device)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.bool, device=device)
        )
        reset_mask = (
            torch.zeros(batch_size, dtype=torch.bool, device=device)
            if reset_mask is None
            else torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
        )
        update_mask = (
            active_mask
            if update_mask is None
            else torch.as_tensor(update_mask, dtype=torch.bool, device=device) & active_mask
        )

        state_before = memory_state
        policy_tokens = qwen.embodied_action_tokens
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.memory_enabled and has_actions:
            if self.memory_module is None or self.policy_memory_fusion is None:
                raise RuntimeError("memory is enabled but its modules were not constructed")
            if state_before is None:
                state_before = self.memory_module.init_state(batch_size, device=device)
            state_before = self.memory_module.reset_state(state_before, reset_mask)
            memory_read = self.memory_module.read(qwen.action_tokens, state_before, read_mask=active_mask)
            diagnostics.update(memory_read.diagnostics)
            if self.memory_schema_version >= 2:
                policy_tokens = self.policy_memory_fusion(
                    policy_tokens, state_before, bypass=fusion_bypass
                )
                diagnostics["policy_gate"] = torch.tanh(self.policy_memory_fusion.gamma_c).detach()
            else:
                policy_tokens = self.policy_memory_fusion(
                    policy_tokens, memory_read.tokens, bypass=fusion_bypass
                )
                diagnostics["policy_gate"] = torch.tanh(self.policy_memory_fusion.gate).detach()

        losses: Dict[str, torch.Tensor] = {}
        scaled_world_loss = None
        if include_world_loss and not masked:
            world_loss = self._compute_world_loss(
                [example["video"] for example in inputs], qwen.action_tokens
            )
            scaled_world_loss = world_loss if not has_actions else world_loss * 0.1

        if has_actions and include_action_loss:
            if policy_tokens is None:
                raise RuntimeError("robot samples require embodied-action tokens")
            losses["action_loss"] = self._compute_action_loss(
                [example["action"] for example in examples],
                [example["state"] for example in examples] if "state" in examples[0] else None,
                policy_tokens,
            )
        if scaled_world_loss is not None:
            losses["wm_loss"] = scaled_world_loss
        if masked:
            losses["rec_loss"] = self._compute_recon_loss(
                [example["video_clean"] for example in examples],
                policy_tokens,
                qwen.action_tokens,
            )

        state_after = state_before
        # The writer is deliberately invoked only after all requested prediction/loss
        # tensors have been formed.  It consumes current Qwen markers, never targets.
        if self.memory_enabled and has_actions:
            if self.memory_schema_version >= 2:
                write_mask = torch.zeros_like(update_mask) if masked else update_mask
                state_after = self.memory_module.write(
                    qwen.action_tokens, state_before, update_mask=write_mask, count_mask=update_mask
                )
            else:
                state_after = self.memory_module.write(
                    qwen.action_tokens, state_before, update_mask=update_mask
                )
        return losses, state_after, diagnostics

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        if examples and isinstance(examples[0], dict) and "steps" in examples[0]:
            return self.forward_sequence(examples)
        if (
            self.memory_schema_version == 3
            and examples
            and isinstance(examples[0], dict)
            and "action" not in examples[0]
        ):
            return self._forward_retro_video(examples)
        losses, _, diagnostics = self._forward_one(examples)
        self.last_memory_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        } if diagnostics else None
        return losses

    def forward_sequence(self, segments: List[dict]) -> Dict[str, torch.Tensor]:
        """Unroll one fully-valid supervised robot segment per batch row."""
        if not self.memory_enabled or self.memory_module is None:
            raise RuntimeError("forward_sequence requires framework.memory.enabled=true")
        if not segments:
            raise ValueError("segments must be non-empty")
        sequence_length = len(segments[0]["steps"])
        if any(len(segment["steps"]) != sequence_length for segment in segments):
            raise ValueError("all segments in a batch must have the same padded length")

        batch_size = len(segments)
        memory_device = next(self.memory_module.parameters()).device
        memory_state = self.memory_module.init_state(batch_size, device=memory_device)
        action_losses: List[torch.Tensor] = []
        world_losses: List[torch.Tensor] = []
        recon_losses: List[torch.Tensor] = []
        nce_anchors: List[torch.Tensor] = []
        last_diagnostics: Dict[str, torch.Tensor] = {}
        include_robot_world = bool(self.config.trainer.get("robot_world_model_loss", True))
        bptt_steps = max(1, int(self.config.trainer.get("memory_bptt_steps", 4)))
        detach_burn_in = bool(self.config.trainer.get("memory_detach_burn_in", True))
        supervised_in_window = 0
        supervised_index = 0
        entered_supervised = False
        had_burn_in_update = False

        mask_plans = [
            np.asarray(segment["mask_plan"], dtype=bool) if segment.get("mask_plan") is not None else None
            for segment in segments
        ]
        if self.memory_schema_version < 2 and any(
            plan is not None and bool(plan.any()) for plan in mask_plans
        ):
            raise ValueError("mask_plan requires framework.memory.schema_version >= 2")

        carriers = []
        for segment in segments:
            carrier = next((step for step in segment["steps"] if step is not None), None)
            if carrier is None:
                raise ValueError("segment contains no real decisions")
            carriers.append(carrier)

        for time_index in range(sequence_length):
            active = torch.as_tensor(
                [segment["sequence_valid"][time_index] for segment in segments],
                dtype=torch.bool,
                device=memory_device,
            )
            if not bool(active.any()):
                continue
            loss_mask = torch.as_tensor(
                [segment["loss_mask"][time_index] for segment in segments],
                dtype=torch.bool,
                device=memory_device,
            )
            update_mask = torch.as_tensor(
                [segment["update_mask"][time_index] for segment in segments],
                dtype=torch.bool,
                device=memory_device,
            )
            reset_mask = torch.as_tensor(
                [segment["is_first"][time_index] for segment in segments],
                dtype=torch.bool,
                device=memory_device,
            ) & active
            step_examples = [
                segment["steps"][time_index]
                if segment["steps"][time_index] is not None
                else carriers[row]
                for row, segment in enumerate(segments)
            ]
            supervised = bool(loss_mask.any())
            if supervised and not bool(torch.all(loss_mask & active)):
                raise ValueError(
                    "Phase-1 action loss is scalar; every row must be valid at supervised timesteps"
                )
            masked = False
            if supervised:
                flags = [
                    bool(plan[supervised_index]) if plan is not None else False
                    for plan in mask_plans
                ]
                if any(flags) and not all(flags):
                    raise ValueError(
                        "mask_plan must agree across batch rows at each supervised decision"
                    )
                masked = flags[0]
                supervised_index += 1
                if not entered_supervised and detach_burn_in and had_burn_in_update:
                    memory_state = memory_state.detach()
                elif supervised_in_window >= bptt_steps:
                    memory_state = memory_state.detach()
                    supervised_in_window = 0
                entered_supervised = True
            losses, memory_state, last_diagnostics = self._forward_one(
                step_examples,
                memory_state=memory_state,
                reset_mask=reset_mask,
                active_mask=active,
                update_mask=update_mask,
                include_action_loss=supervised,
                include_world_loss=supervised and include_robot_world,
                masked=masked,
            )
            if supervised:
                supervised_in_window += 1
                action_losses.append(losses["action_loss"])
                if "wm_loss" in losses:
                    world_losses.append(losses["wm_loss"])
                if "rec_loss" in losses:
                    recon_losses.append(losses["rec_loss"])
                if (
                    self.memory_schema_version >= 2
                    and self.policy_memory_fusion.last_residual is not None
                ):
                    nce_anchors.append(
                        F.normalize(
                            self.nce_head_h(self.policy_memory_fusion.last_residual.mean(dim=1)),
                            dim=-1,
                        )
                    )
            elif bool(update_mask.any()):
                had_burn_in_update = True

        if not action_losses:
            raise ValueError("segment batch has no supervised decisions")
        output = {"action_loss": torch.stack(action_losses).mean()}
        if world_losses:
            output["wm_loss"] = torch.stack(world_losses).mean()
        if recon_losses:
            output["rec_loss"] = torch.stack(recon_losses).mean()
        if nce_anchors:
            positive = self._nce_positive(segments, nce_anchors[0])
            output["nce_anchor"] = torch.cat(nce_anchors, dim=0)
            output["nce_positive"] = positive.repeat(len(nce_anchors), 1)
        self.last_memory_diagnostics = {
            key: value.detach() for key, value in last_diagnostics.items()
        } if last_diagnostics else None
        return output

    def _nce_positive(self, segments: List[dict], reference: torch.Tensor) -> torch.Tensor:
        """Frozen-teacher positive from one unmasked past decision per row.

        The source is the decision four before the first supervised decision
        when the burn-in reaches that far, else the row's earliest valid
        decision; both are structurally unmasked.
        """
        sources = []
        for segment in segments:
            valid = np.asarray(segment["sequence_valid"], dtype=bool)
            first_supervised = int(np.flatnonzero(np.asarray(segment["loss_mask"], dtype=bool))[0])
            source = first_supervised - 4
            if source < 0 or not valid[source]:
                source = int(np.flatnonzero(valid)[0])
            sources.append(segment["steps"][source])
        latents, tokens_per_frame, _ = self._encode_video_latents(
            [example["video"] for example in sources], reference
        )
        pooled = latents[:, tokens_per_frame:, :].mean(dim=1)
        return F.normalize(self.nce_head_g(pooled), dim=-1)

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        memory_state: Optional[MemoryState] = None,
        reset_mask=None,
        return_memory_state: bool = False,
        generator: Optional[torch.Generator] = None,
        initial_noise: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        memory_bypass: bool = False,
        qwen_cache: Optional[QwenTokenBundle] = None,
        keep_qwen_cache: bool = False,
        **kwargs,
    ):
        if qwen_cache is not None:
            qwen = qwen_cache
        else:
            train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            qwen = self._encode_qwen_tokens(
                batch_images,
                instructions,
                self.config.datasets.vla_data.get("CoT_prompt", ""),
                require_embodied=True,
            )
        # A kept cache lets a paired same-decision call (e.g. the bypass
        # counterfactual) skip the resize + Qwen encode entirely.
        self.last_qwen_cache = qwen if keep_qwen_cache else None
        embodied = qwen.embodied_action_tokens
        if embodied is None:
            raise RuntimeError("inference prompt did not produce embodied-action tokens")
        state_before = memory_state
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.memory_enabled and not memory_bypass:
            batch_size = embodied.shape[0]
            device = embodied.device
            if state_before is None:
                state_before = self.memory_module.init_state(batch_size, device=device)
            reset = (
                torch.zeros(batch_size, dtype=torch.bool, device=device)
                if reset_mask is None
                else torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
            )
            state_before = self.memory_module.reset_state(state_before, reset)
            memory_read = self.memory_module.read(qwen.action_tokens, state_before)
            diagnostics.update(memory_read.diagnostics)
            if self.memory_schema_version >= 2:
                embodied = self.policy_memory_fusion(embodied, state_before)
            else:
                embodied = self.policy_memory_fusion(embodied, memory_read.tokens)

        proprio = (
            torch.as_tensor(np.array(state), device=embodied.device, dtype=embodied.dtype)
            if state is not None
            else None
        )
        with self._autocast_context(embodied, torch.bfloat16):
            pred_actions = self.action_model.predict_action(
                embodied,
                proprio,
                generator=generator,
                initial_noise=initial_noise,
            )

        state_after = state_before
        if self.memory_enabled and update_memory and not memory_bypass:
            state_after = self.memory_module.write(qwen.action_tokens, state_before)
        if state_after is not None:
            state_after = state_after.detach()
        self.last_memory_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        } if diagnostics else None
        public_output = {
            "normalized_actions": pred_actions.to(
                dtype=torch.float32
            ).detach().cpu().numpy(),
            "embodied_action_tokens": qwen.embodied_action_tokens.to(
                dtype=torch.float32
            ).detach().cpu().numpy(),
        }
        if return_memory_state:
            return public_output, state_after
        return public_output



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
