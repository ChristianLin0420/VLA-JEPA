## MAP REPORT: architecture

# VLA-JEPA Core Model Architecture Report

Repo root: `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA` (all paths below relative to this unless absolute).

## 1. Framework class and composition

- Main class: `VLA_JEPA(baseframework)`, registered as `"VLA_JEPA"` via `@FRAMEWORK_REGISTRY.register("VLA_JEPA")` — `starVLA/model/framework/VLA_JEPA.py:44-45`. Built by `build_framework(cfg)` from `cfg.framework.name` — `starVLA/model/framework/__init__.py:35-61`. Base class `baseframework(PreTrainedModel)` provides `from_pretrained` (strict state_dict load + norm_stats attach) and `unnormalize_actions` — `starVLA/model/framework/base_framework.py:37-114, 175-205`. Other frameworks in the dir (`M1.py`, `QwenGR00T.py`, `QwenOFT.py`, `QwenFast.py`, `QwenPI.py`, `QwenDual.py`) are siblings from starVLA; the memv1 runs use only `VLA_JEPA`.

Submodules constructed in `__init__` (`VLA_JEPA.py:57-167`):
1. **Qwen-VL backbone** `self.qwen_vl_interface = get_vlm_model(config)` (`VLA_JEPA.py:71`). For `base_vlm` containing "Qwen3-VL" this is `_QWen3_VL_Interface` wrapping `Qwen3VLForConditionalGeneration` loaded in bf16 (`starVLA/model/modules/vlm/__init__.py:11-14`, `starVLA/model/modules/vlm/QWen3.py:63-76`). Qwen3-VL-2B-Instruct: text `hidden_size=2048`, 28 layers (model config at `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/models/Qwen3-VL-2B-Instruct/config.json`); `model.config.hidden_size` aliased to `text_config.hidden_size` at `QWen3.py:76`.
2. **Tokenizer expansion** `expand_tokenizer` adds `action_horizon*4 = 28` special tokens `<|action_0|>..<|action_27|>` plus `<|embodied_action|>`, and resizes Qwen embeddings (`VLA_JEPA.py:72-78, 189-216`).
3. **Action head** `self.action_model = FlowmatchingActionHead` (GR00T N1.5 flow-matching DiT) (`VLA_JEPA.py:85`); its `diffusion_model_cfg.cross_attention_dim` is force-set to the Qwen hidden size (2048) at `VLA_JEPA.py:83`.
4. **JEPA target encoder** `self.vj_encoder = AutoModel.from_pretrained(framework.vj2_model.base_encoder)` = V-JEPA2 ViT-L (`vjepa2-vitl-fpc64-256`: hidden 1024, image 256, patch 16, tubelet 2, 24 layers) + `self.vj_processor` (`VLA_JEPA.py:91-92`).
5. **JEPA predictor** `self.vj_predictor = VisionTransformerPredictorAC(num_frames=8//2=4, img_size=(256,256), tubelet_size=1, depth=12, num_heads=8, embed_dim=1024*2=2048 (two-view concat), predictor_embed_dim=1024 (default), action_embed_dim=2048 (Qwen hidden), num_add_tokens=8)` (`VLA_JEPA.py:94-104`; class in `starVLA/model/modules/world_model/vj2_predictor.py:17-119`).
6. **Prompt marker strings**: `replace_prompt` = first `(8/2 − 1)=3` action tokens each repeated `num_action_tokens_per_timestep=8` times (24 markers total) (`VLA_JEPA.py:105-108`); `embodied_replace_prompt` = `<|embodied_action|>` × 32 (`VLA_JEPA.py:110`). `expected_action_token_count = 24`, `expected_embodied_token_count = 32` (`VLA_JEPA.py:162-167`).
7. **memv1 memory** (only when `framework.memory.enabled`): `self.memory_module = RecurrentMemory(source_dim=2048, num_slots=8, memory_dim=512, num_heads=8, update_gate_init=0.1)` and `self.policy_memory_fusion = ResidualMemoryFusion(consumer_dim=2048, memory_dim=512, bottleneck_dim=512, num_heads=8, gate_init=1e-3)` (`VLA_JEPA.py:124-160`). Phase-2/3 features (`long_term`, `world_model_conditioning`) raise `NotImplementedError` if enabled (`VLA_JEPA.py:137-140`). When memory is disabled the model is checkpoint-identical to the legacy model (`VLA_JEPA.py:122-127`).

## 2. Input modalities and prompt encoding

Robot sample dict from LeRobot mixture (`starVLA/dataloader/gr00t_lerobot/datasets.py:1846-1888`, `_format_step`): `action` [chunk, 7] fp16, `image` = list of 2 PIL images resized to 224×224 (first frame of each view), `lang` string, `video` np.ndarray [n_views=2, T=8, 256, 256, 3] (single-view datasets duplicate the view, `datasets.py:1874-1875`); optional `state` [1, state_dim] only if `with_state: true` (`datasets.py:1882-1887`; all memv1 configs set `with_state: false`). Video-only (SSV2) samples have only `image` (1 view), `video` (duplicated to 2 views), `lang` — `starVLA/dataloader/video_datasets.py:62-76`. `video_horizon` = `framework.vj2_model.num_frames` and `action_horizon` = `framework.action_model.action_horizon` are passed to the dataset builder at `starVLA/dataloader/__init__.py:44-57`.

`_encode_qwen_tokens` (`VLA_JEPA.py:242-282`): builds a chat prompt from `datasets.vla_data.CoT_prompt` ("Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}.") substituting `{actions}` → 24 action markers and `{e_actions}` → 32 embodied markers (`VLA_JEPA.py:249-257`; template application in `QWen3.py:111-157`). One Qwen forward with `output_hidden_states=True` under bf16 autocast; `last_hidden = outputs.hidden_states[-1]` shape **[B, L, 2048]**. `_select_token_rows` (`VLA_JEPA.py:224-240`) extracts:
- `action_tokens` **[B, 24, 2048]** (rows at the 24 `<|action_i|>` positions; hard count check raises on mismatch),
- `embodied_action_tokens` **[B, 32, 2048]** (robot samples only).

## 3. Forward pass / losses

`forward` (`VLA_JEPA.py:419-426`) dispatches: if `examples[0]` contains `"steps"` → `forward_sequence` (memory segment unroll), else `_forward_one`.

`_forward_one` (`VLA_JEPA.py:337-417`):
- Batch must be homogeneous robot vs video-only (`VLA_JEPA.py:349-351`); prompt selected from `datasets.vla_data.CoT_prompt` (robot) or `datasets.video_data.CoT_prompt` (video) (`VLA_JEPA.py:355-359`).
- **Memory read + fusion** (robot batches, memory enabled): `memory_read = memory_module.read(qwen.action_tokens, state)` → tokens **[B, 8, 512]** fp32; `policy_tokens = policy_memory_fusion(embodied_tokens, memory_read.tokens)` = `embodied + tanh(gate)·CrossAttn(2048→512→2048)` residual (`VLA_JEPA.py:382-391`; `starVLA/model/modules/memory/fusion.py:86-100`; read at `recurrent_memory.py:201-229`). Memory **write** happens only after all losses are formed, from current-step Qwen action markers (never targets): gated attention update `working = (1−σ(gate))·prev + σ(gate)·tanh(candidate)`, FP32, autocast disabled (`VLA_JEPA.py:412-416`, `recurrent_memory.py:231-279`). State type `MemoryState(working [B,8,512] fp32, episodic=None, steps [B] i64, valid [B] bool)` — `starVLA/model/modules/memory/state.py:14-87`.
- **World (JEPA) loss** `_compute_world_loss` (`VLA_JEPA.py:284-310`), see §4. Scaling: `world_loss * 0.1` for robot batches, `* 1.0` for video-only batches — hardcoded at `VLA_JEPA.py:399` (the `trainer.loss_scale` yaml block is NOT read by `train_vlajepa_cotrain.py`; it is only used in the older `train_starvla_cotrain.py:387`).
- **Action loss** `_compute_action_loss` (`VLA_JEPA.py:312-335`), see §5.
- Returns `{"action_loss", "wm_loss"}`; trainer sums them (`train_vlajepa_cotrain.py:427`).

`forward_sequence` (`VLA_JEPA.py:428-524`) — memory-enabled segment unroll: each batch row is a segment dict (`steps` list of length `burn_in_max_decisions(8) + segment_length(4) = 12`, plus `sequence_valid/loss_mask/update_mask/is_first` masks, built in `datasets.py:1719-1795`). Per timestep it calls `_forward_one` with masks; burn-in steps run without loss (memory write only); supervised steps (last K=4) accrue `action_loss` and optionally `wm_loss` (gated by `trainer.robot_world_model_loss`, `VLA_JEPA.py:444`). Truncated BPTT: state detached after `trainer.memory_bptt_steps` (default 4) supervised steps and optionally at the burn-in→supervised boundary (`trainer.memory_detach_burn_in`) (`VLA_JEPA.py:445-497`). Output is the mean over supervised steps (`VLA_JEPA.py:516-520`).

## 4. JEPA prediction objective

`_compute_world_loss` (`VLA_JEPA.py:284-310`):
- videos stacked to [B, V=2, T=8, C, H=256, W=256] (`:289-291`), per-clip processed by V-JEPA2 processor, concatenated → `input_videos` **[B·2, 8, 3, 256, 256]**.
- **Target/context encoder = frozen V-JEPA2, wrapped in `torch.no_grad()`** (`:300-302`): `vj_encoder.get_vision_features` → **[B·2, 1024, 1024]** (tokens = (8/tubelet 2)·(256/16)² = 4·256 = 1024); views chunked and concatenated on channel dim → **[B, 1024, 2048]** (`:302`). There is no EMA — the same frozen encoder provides both context features and prediction targets.
- Temporal shift: `tokens_per_frame = 1024/4 = 256`; `input_states = emb[:, :768, :]` (latent frames 0-2), `gt_states = emb[:, 256:, :]` (latent frames 1-3) (`:303-306`).
- **Predictor** `vj_predictor(input_states, action_tokens)` (`:307`): maps 2048→1024 (`vj2_predictor.py:144`), encodes Qwen action-marker tokens 2048→1024 and interleaves 8 action tokens before each frame's 256 patch tokens → sequence **[B, 3·264=792, 1024]** (`vj2_predictor.py:151-160`); 12 `ACBlock`s with a block-causal frame-level attention mask (`build_action_block_causal_attention_mask`, `vj2_modules.py:12-23`, mask sliced at `vj2_predictor.py:162`) and 3D RoPE on frame tokens / temporal-only RoPE on action tokens (`vj2_modules.py:168-246`); action tokens sliced off and output projected back 1024→2048 → **predicted_states [B, 768, 2048]** (`vj2_predictor.py:189-196`).
- **Loss = `F.l1_loss(predicted_states, gt_states, mean)`** (`VLA_JEPA.py:308`). Not cosine — cosine similarity (`jepa/pred_gt_cosine_*`), identity-baseline cosine, and collapse spectra are computed *only as logging diagnostics* from captured detached tensors (`starVLA/training/trainer_utils/jepa_analysis.py:102-166`; capture side-channel `VLA_JEPA.py:112-120, 169-187`; trainer toggles `capture_jepa` at `train_vlajepa_cotrain.py:423-430` and logs at `:490-513`).

## 5. Action chunking / flow-matching head

`FlowmatchingActionHead` (`starVLA/model/modules/action_model/GR00T_ActionHeader.py:216-398`):
- DiT-B: `input_embedding_dim=768`, 12 heads × 64 = inner_dim 768 (`GR00T_ActionHeader.py:211-214`), 16 layers with interleaved self/cross attention (odd layers self-attn, even layers cross-attn onto `encoder_hidden_states`), ada_norm timestep conditioning, output proj 768→`output_dim=1024` (`cross_attention_dit.py:191-308`). `action_decoder` MLP 1024→1024→7 (`GR00T_ActionHeader.py:246-250`); `action_encoder` embeds noisy actions+timestep 7→768 (`:64-105, 242-245`); `state_encoder` MLP `state_dim(8)→1024→768` built when `state_dim` set (`:236-240`); `future_tokens` embedding [32, 768] (`:251-252`); learned positional embedding [1024, 768] when `add_pos_embed` (`:254-256`).
- **Chunking**: `action_horizon = future_action_window_size + 1 = 7` (`GR00T_ActionHeader.py:233`); training target `actions[:, -(future_action_window_size+1):, :]` → **[B, 7, 7]** (`VLA_JEPA.py:321`); `chunk_len = past(0)+1+future(6) = 7` (`VLA_JEPA.py:87-89`).
- **Training** (`_compute_action_loss`, `VLA_JEPA.py:312-335` → head `forward` `GR00T_ActionHeader.py:270-317`): actions and conditioning repeated `trainer.repeated_diffusion_steps=4`× → actions **[4B, 7, 7]**, conditioning `vl_embs` = memory-fused embodied tokens **[4B, 32, 2048]**. Flow matching: `t ~ (s − Beta(1.5, 1.0))/s` with `s=0.999` (`:262-264`), `noisy = (1−t)·noise + t·actions`, target `velocity = actions − noise`, t discretized into `num_timestep_buckets=1000`; DiT input `sa_embs = cat(state?, future_tokens[32], action_features[7])` → **[4B, 39, 768]** (no state in memv1 runs); loss = MSE(pred_velocity, velocity) (`:316`). Explicit bf16 autocast around the head (`VLA_JEPA.py:333-335`).
- **Inference** `predict_action` (framework: `VLA_JEPA.py:526-599`; head: `GR00T_ActionHeader.py:319-390`): images resized to `datasets.vla_data.image_size` if set; Qwen encode with `require_embodied=True`; optional memory read/fuse and (if `update_memory`) post-prediction write (`VLA_JEPA.py:555-568, 583-587`); Euler integration of the flow from noise **[B, 7, 7]** for `num_inference_timesteps=4` steps (`GR00T_ActionHeader.py:351-389`); returns `normalized_actions` (numpy) + raw embodied tokens (`VLA_JEPA.py:591-596`). Un-normalization is the caller's job via `baseframework.unnormalize_actions` (clip to [-1,1]; channel 6 binarized at 0.5; q01/q99 linear rescale) — `base_framework.py:175-205`.

## 6. Frozen vs trainable in co-training

Freezing is trainer-driven: `TrainerUtils.freeze_backbones` sets `requires_grad=False` for comma-separated module paths in `trainer.freeze_modules` (`starVLA/training/trainer_utils/trainer_tools.py:157-199`, invoked at `train_vlajepa_cotrain.py:203`); `build_param_lr_groups` excludes frozen params and builds per-module LR groups from `trainer.learning_rate` (`trainer_tools.py:51-113`).

Per run config (repo configs match the deployed run configs in `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/*/config.yaml` up to YAML formatting):
- **vlajepa_memv1_video** (`scripts/config/vlajepa_memv1_video.yaml:114`): `freeze_modules: "vj_encoder,action_model"`; trainable = Qwen (LR 1e-5) + vj_predictor (base LR 3e-5); memory disabled (`:57`). World-model-only pretrain via `starVLA/training/train_vlajepa_video.py`.
- **vlajepa_memv1_stage1** (`scripts/config/vlajepa_memv1_stage1.yaml:161`): `freeze_modules: "qwen_vl_interface,vj_encoder,vj_predictor"`; trainable = action_model (1e-4), memory_module (1e-4), policy_memory_fusion (1e-4); `robot_only: true`, `skip_video_pass: true`, `robot_world_model_loss: false` (`:129-131`); checkpoint migration allows missing `memory_module.`/`policy_memory_fusion.` prefixes (`:145-151`).
- **vlajepa_memv1_cotrain** (`scripts/config/vlajepa_memv1_cotrain.yaml:160`): `freeze_modules: "qwen_vl_interface,vj_encoder"`; trainable = **vj_predictor** (falls into base group, LR 3e-5), **action_model** (1e-4), **memory_module** (1e-4), **policy_memory_fusion** (1e-4). Comment in config: Qwen frozen to keep K=4 unrolling in the H100 envelope (`:158-159`).
- Independent of `freeze_modules`, `vj_encoder` is always a no-grad target encoder inside the forward (`VLA_JEPA.py:300-302`), and DDP runs with `find_unused_parameters=True` because of it and the video pass skipping the action head (`train_vlajepa_cotrain.py:54-58`).
- Cotrain step = two optimizer updates: VLA pass (action_loss + 0.1·wm_loss) then video pass (wm_loss only), each with own zero_grad/backward/clip/step/scheduler-step (`train_vlajepa_cotrain.py:418-452`); LR schedule stretched by `trainer.optimizer_steps_per_training_step` (2 for cotrain, 1 for stage1) (`train_vlajepa_cotrain.py:129-145`).

## 7. Config knobs (memv1 values; consumer locations)

`framework.qwenvl`: `base_vlm` (model path; also selects wrapper class, `vlm/__init__.py:6-14`), `attn_implementation` (sdpa; `QWen3.py:61`), `vl_hidden_dim` (2048 — **not read** by the VLA_JEPA path; real dim taken from Qwen config at `VLA_JEPA.py:83,102,142`).

`framework.action_model` (consumed in `GR00T_ActionHeader.py:222-260` and `VLA_JEPA.py:76,83,87-89`): `action_model_type: DiT-B` (selects `input_embedding_dim=768/heads` table `GR00T_ActionHeader.py:211-214`), `hidden_size: 1024` (decoder MLP width), `action_hidden_dim` (unused by this head — only other headers, e.g. `DiTActionHeader.py:210`), `add_pos_embed: true`, `max_seq_len: 1024` (pos-emb table), `action_dim: 7`, `state_dim: 8`, `future_action_window_size: 6`, `action_horizon: 7` (dataset chunk + tokenizer sizing `VLA_JEPA.py:76`, `dataloader/__init__.py:46`), `past_action_window_size: 0` (`VLA_JEPA.py:88`), `noise_beta_alpha: 1.5`/`noise_beta_beta: 1.0`/`noise_s: 0.999` (`GR00T_ActionHeader.py:258-264`), `num_timestep_buckets: 1000`, `num_inference_timesteps: 4`, `num_target_vision_tokens: 32` (`:251`), `repeated_diffusion_steps` (framework-level copy unused; the effective one is `trainer.repeated_diffusion_steps`, read at `VLA_JEPA.py:324`), `diffusion_model_cfg`: `cross_attention_dim: 2048` (overwritten anyway `VLA_JEPA.py:83`), `dropout: 0.2`, `final_dropout: true`, `interleave_self_attention: true`, `norm_type: ada_norm`, `num_layers: 16`, `output_dim: 1024`, `positional_embeddings: null` (all into `DiT.__init__`, `cross_attention_dit.py:196-254`).

`framework.vj2_model` (consumed `VLA_JEPA.py:72-110, 162-167`): `base_encoder` (V-JEPA2 path), `depth: 12` and `num_heads: 8` (predictor), `special_action_token: "<|action_{}|>"`, `num_action_tokens_per_timestep: 8`, `embodied_action_token: "<|embodied_action|>"`, `num_embodied_action_tokens_per_instruction: 32`, `num_frames: 8` (also sets dataset `video_horizon`, `dataloader/__init__.py:47,115`).

`framework.memory` (consumed `VLA_JEPA.py:124-160`): `enabled`, `schema_version: 1`, `source`/`read_before_write`/`state_dtype` (documentary; not read in code), `short_term.{enabled, num_slots: 8, dim: 512, num_heads: 8, update_gate_init: 0.1}`, `action_conditioning.{enabled, bottleneck_dim: 512, num_heads, dropout: 0.0, zero_init_gate: false, gate_init: 1e-3}`, `long_term.enabled=false` and `world_model_conditioning.enabled=false` (must stay false, `VLA_JEPA.py:137-140`).

`trainer` knobs touching the model: `repeated_diffusion_steps: 4` (`VLA_JEPA.py:324`), `robot_world_model_loss` (`VLA_JEPA.py:444`), `memory_bptt_steps: 4` (`VLA_JEPA.py:445`), `memory_detach_burn_in: true` (`VLA_JEPA.py:446`), `freeze_modules`, `learning_rate.{base, qwen_vl_interface, action_model, memory_module, policy_memory_fusion}`, `robot_only` (`train_vlajepa_cotrain.py:108`), `optimizer_steps_per_training_step`, `pretrained_checkpoint` + `checkpoint_migration.{enabled, allow_missing_prefixes}` (`train_vlajepa_cotrain.py:79-89,196-202`), `jepa_log_interval`/`jepa_figure_interval` (`:172-173`). `skip_video_pass` and `memory_direct_context_dropout` appear in configs but are **not referenced anywhere in `starVLA/**/*.py`** (grep-verified; video pass is actually disabled by `robot_only`).

`datasets.vla_data` knobs used by the model/dataloader: `CoT_prompt` (`VLA_JEPA.py:356-358,547`), `image_size` (inference resize, `VLA_JEPA.py:540-542`), `resolution_size: 224` (VLM image), `video_resolution_size: 256` (world-model clip), `with_state: false`, `sample_mode: contiguous_segment`, `segment_length: 4`, `burn_in_max_decisions: 8`, `segment_stride: 7`, `per_device_batch_size: 1` (segments) (`starVLA/dataloader/lerobot_datasets.py:54-134`, `datasets.py:1746-1795`). `datasets.video_data.CoT_prompt` for the SSV2 pass (`VLA_JEPA.py:358`).

## 8. Key tensor shape summary (memv1 config, batch B, 2 views)

| Tensor | Shape | Source |
|---|---|---|
| Qwen last hidden | [B, L, 2048] | `VLA_JEPA.py:265` |
| action-marker tokens | [B, 24, 2048] | `VLA_JEPA.py:266-272` |
| embodied tokens | [B, 32, 2048] | `VLA_JEPA.py:273-281` |
| V-JEPA2 features per view | [B·2, 1024, 1024] | `VLA_JEPA.py:301` |
| multi-view latent | [B, 1024, 2048] | `VLA_JEPA.py:302` |
| input_states / gt_states / predicted | [B, 768, 2048] | `VLA_JEPA.py:305-307` |
| predictor internal seq | [B, 792, 1024] | `vj2_predictor.py:144-160` |
| memory working state / read | [B, 8, 512] fp32 | `state.py:29`, `recurrent_memory.py:221-229` |
| action target (train) | [4B, 7, 7] after ×4 repeat | `VLA_JEPA.py:321-326` |
| DiT input tokens | [4B, 39, 768] (32 future + 7 action) | `GR00T_ActionHeader.py:301-303` |
| predicted action chunk (inference) | [B, 7, 7] normalized | `GR00T_ActionHeader.py:336, 389` |

## MAP REPORT: memory

# memv1 Memory Module — Full Report

Path shorthand (defined once, all references below expand to absolute paths):
- `REPO` = `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA`
- `RUNS` = `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs`

## 1. Implementation files

- `REPO/starVLA/model/modules/memory/state.py` — `MemoryState`, `MemoryRead` dataclasses.
- `REPO/starVLA/model/modules/memory/recurrent_memory.py` — `RecurrentMemory` (the `memory_module`).
- `REPO/starVLA/model/modules/memory/fusion.py` — `ResidualMemoryFusion` (the `policy_memory_fusion`).
- Integration: `REPO/starVLA/model/framework/VLA_JEPA.py` (construction 124–160; train path `_forward_one` 337–417; `forward_sequence` 428–524; inference `predict_action` 526–599).
- Serving: `REPO/deployment/model_server/server_policy.py`, `REPO/deployment/model_server/tools/websocket_policy_server.py`.

## 2. Memory state structure

`MemoryState` (state.py:14–87), frozen dataclass, activations only ("an episode's state must never be registered as a model parameter or buffer", state.py:1–6):
- `working`: `[B, S=8 slots, D=512]` **FP32 enforced** (state.py:37–38); slots=8, dim=512 from config `framework.memory.short_term.{num_slots,dim}` (`REPO/scripts/config/vlajepa_memv1_stage1.yaml:78–79`), instantiated at VLA_JEPA.py:144–150.
- `episodic`: `Optional [B, Dv, Dk]` FP32 — **always None in Phase 1**; reserved for the Phase-2 gated-delta tier (state.py:20–23, 51–57).
- `steps`: `[B]` int64 count of completed writes (state.py:23–24).
- `valid`: `[B]` bool active-row mask (state.py:25–27).
- `detach()` / `to(device=)` preserve dtypes (state.py:69–87). `MemoryRead` = `{tokens [B,S,D] FP32, diagnostics dict}` (state.py:90–103).
- Runtime state never appears in `state_dict()`; export refuses if `memory_state`/`last_memory` keys appear (`REPO/cluster/export_vlajepa_ckpt.py:50–52`).

Learned parameters (31 tensors, all FP32; verified in `RUNS/vlajepa_memv1_cotrain/checkpoints/step_34729/model.safetensors`): `memory_module.*` = 2,900,480 params; `policy_memory_fusion.*` = 3,418,113 params; total ≈ 6.32M (cap test `<10M` at `REPO/tests/memory/test_memory_fusion.py:80–85`).

## 3. Write/update rule (recurrent_memory.py:231–279)

Source = the **24 current Qwen action-marker hidden states** `[B,24,2048]` (`qwen.action_tokens`; count = `(num_frames/tubelet 8/2 − 1) × 8 tokens/timestep` at VLA_JEPA.py:162–164). Config label: `source: qwen_action_tokens_current_only` (stage1 yaml:59; declarative only, string never read in code).

Under `torch.autocast(enabled=False)` in FP32 (recurrent_memory.py:252):
1. `source = source_projection(source_norm(tokens_fp32))` 2048→512 (line 254).
2. Query = `slot_norm(previous + slot_ids)`; `slot_ids` are learned per-slot identity embeddings `[8,512]` to prevent slot collapse (lines 46, 257–258).
3. `context = MultiheadAttention(query, key=source, value=source)` — 8 heads, dim 512, batch_first (lines 50–55, 259–264).
4. `gate = sigmoid(update_gate(cat(slot_norm(previous), context)))` — Linear 1024→512, per-slot per-channel gate (lines 56, 266–267).
5. `candidate = tanh(candidate_projection(context))`; `proposed = (1−gate)·previous + gate·candidate` (lines 268–269) — convex GRU-style update, so the state is bounded.
6. Row masking: `working = where(active, proposed, previous)`; `steps += active` (lines 271–272). `active = update_mask & state.valid` (lines 241–249).

Update-gate init: `update_gate.weight = 0`, `bias = logit(update_gate_init=0.1)` so the initial update probability is exactly 0.1 (lines 69–73; config `short_term.update_gate_init: 0.1`, stage1 yaml:82).

## 4. Read rule (recurrent_memory.py:201–229)

`read()` returns the **previous** working slots unmodified (read-before-write; write is invoked only after all loss/prediction tensors are formed — VLA_JEPA.py:412–416, and at inference after the action head runs — VLA_JEPA.py:583–585). Inactive rows read zeros (lines 221–223). Diagnostics: `working_norm` (mean slot L2), `steps`, `active` (lines 224–228).

## 5. Fusion into the policy (fusion.py:7–100; VLA_JEPA.py:388–391, 566–568)

Consumer = the **32 embodied-action tokens** `[B,32,2048]` that condition the DiT flow-matching action head (fusion happens once, before the ×4 `repeated_diffusion_steps` expansion — VLA_JEPA.py:322–326 comment). FP32, autocast disabled (fusion.py:86):
- `query = query_projection(consumer_norm(consumer))` 2048→512; `kv = memory_projection(memory_norm(memory))` 512→512; MHA 8 heads at bottleneck 512; `residual = output_projection(attended)` 512→2048 (fusion.py:89–97).
- `output = consumer + tanh(gate) · residual` (fusion.py:98–100).

**Gate**: a single scalar `nn.Parameter` (fusion.py:42); effective gate = `tanh(gate)`. Init: config `action_conditioning.zero_init_gate: false` + `gate_init: 1.0e-03` (stage1 yaml:67–68) → constructed with `gate_init=1e-3` (VLA_JEPA.py:151–152, 159); if `zero_init_gate: true` it would be exactly 0. All projections Xavier-uniform, zero bias (fusion.py:46–53).

**Measured gate trajectory** (verified from checkpoints):
- stage1 step_5000: `policy_memory_fusion.gate = 0.0125118`, tanh = **0.0125111** (`RUNS/vlajepa_memv1_stage1/checkpoints/step_5000/model.safetensors`); matches wandb summary `memory/policy_gate: 0.012511339` at `_step: 5000` (`RUNS/vlajepa_memv1_stage1/wandb/wandb/run-20260701_201702-*/files/wandb-summary.json`, which also shows `memory/working_norm: 1.4977`, `memory/steps: 11`, `memory/active: 1`, `data/processed_decisions: 160000`).
- cotrain step_34729: `gate = 0.0219509`, tanh = **0.0219474** (`RUNS/vlajepa_memv1_cotrain/checkpoints/step_34729/model.safetensors`). Note: cotrain wandb/tensorboard contain **no** `memory/*` scalars at all — the per-step video pass (`_forward_one` with no actions → empty diagnostics) overwrites `last_memory_diagnostics` to `None` (VLA_JEPA.py:422–426) before the trainer reads it (`REPO/starVLA/training/train_vlajepa_cotrain.py:462–468`), so the 0.022 figure exists only in the checkpoint. Diagnostic gate = `tanh(gate).detach()` logged as `policy_gate` (VLA_JEPA.py:391).

## 6. Learned initial slots and reset

- `initial_slots [8,512]`, normal init std=0.02 (recurrent_memory.py:45, 62). `init_state()` expands them per row, zeros invalid rows, steps=0 (lines 122–151). Reset therefore restores a **learned prior**, not zeros.
- `reset_state(state, reset_mask, valid_mask)` out-of-place: reset rows ← `initial_slots`, steps ← 0, episodic ← 0; unselected rows keep state (lines 153–199). Selective per-row reset supported (tested at `REPO/tests/memory/test_recurrent_memory.py:39`).

## 7. Inference-time evolution within an episode, and episode reset

`predict_action` (VLA_JEPA.py:526–599), `@torch.inference_mode()`: per decision — if `memory_state is None` → `init_state` (559); apply `reset_mask` (560–565, default zeros); `read` (566); fuse into embodied tokens (568); run DiT action head (575–581); then `write` iff `update_memory=True` (583–585); returned state is always `.detach()`ed (586–587); returned separately only when `return_memory_state=True` (597–598) — never in the public output dict.

Server session lifecycle (`websocket_policy_server.py`): per-WebSocket `_ConnectionSession{memory_state, generator, episode_id, batch_size, ready}` (lines 19–27). Memory-enabled inference **requires an explicit `reset` RPC first** (lines 188–190) and enforces exactly one `B=1` episode row per connection (lines 191–198). `reset` sets `memory_state=None` and a fresh `torch.Generator` seeded with `episode_seed` (lines 147–159) — that is the between-episode reset; the next infer re-inits from learned initial slots. On inference failure, both memory state and RNG state roll back (lines 222–226); state commits only on success in live mode (215–217). The LIBERO runner calls `model.reset(...)` per episode (`REPO/examples/LIBERO/eval_libero.py:149`) → `client.reset(episode_id=f"libero-{n}", episode_seed=n)` (`REPO/examples/LIBERO/model2libero_interface.py:70–76`); `step()` also auto-resets on task-description change (:104–106).

## 8. MEMORY_MODE=live vs zero

Read at `REPO/deployment/model_server/server_policy.py:44` (`os.environ.get("MEMORY_MODE", "live")`); validated to `{live, zero}` (websocket_policy_server.py:47–49); plumbed in eval via `REPO/cluster/eval_after_run.sbatch:34,43–44,99` and pipeline `REPO/cluster/submit_vlajepa_memv1_pipeline.sh:126,193` (pipeline eval submits **live** only). Server also re-floats memory modules to FP32 after the `--use_bf16` cast (server_policy.py:28–32).

Semantics (websocket_policy_server.py:208–217):
- **live**: `memory_state=state_before, update_memory=True`, candidate state committed → normal recurrence.
- **zero**: `memory_state=None` every call, `update_memory=False`, candidate never committed (test `test_zero_mode_never_commits_state`, `REPO/tests/memory/test_websocket_memory.py:116–122`).

**Why 'zero' is NOT a true no-memory bypass**: with `memory_state=None`, `predict_action` re-runs `init_state` (VLA_JEPA.py:558–559), reads the **trained non-zero `initial_slots`** (recurrent_memory.py:143–144, 221–223), and still executes the full fusion `embodied + tanh(gate)·output_projection(attention(...))` with the trained gate ≈0.022 (VLA_JEPA.py:568; fusion.py:89–100). So 'zero' = "reset-before-every-decision with no writes + learned-prior injection", i.e. a constant-state conditioning path, not the eval-plan definition of `zero` ("injected residual is forced exactly to zero", `REPO/docs/VLA-JEPA-Memory-Evaluation-Plan.md:37`) and not `runtime_bypass` ("skip memory read/write and fusion entirely", same doc :35). The only exact-bypass mechanisms are `ResidualMemoryFusion.forward(..., bypass=True)` (fusion.py:60, 82–83) — **never invoked by any model/server code**, only by the unit test (`REPO/tests/memory/test_memory_fusion.py:65–74`) — or building with `memory.enabled: false` (different module set/checkpoint keys, VLA_JEPA.py:124–129). Exported eval checkpoints: `RUNS/vlajepa_memv1_cotrain/checkpoints/VLA-JEPA-memv1-zero-step_34729.pt` is a **symlink** to `VLA-JEPA-memv1-live-step_34729.pt` (identical weights; mode is purely a serve-time env var).

## 9. Training pipeline and Stage 1

Pipeline chain (submit_vlajepa_memv1_pipeline.sh:12–27, 160–195): `vlajepa_memv1_video` (50K steps, **memory disabled** — `REPO/scripts/config/vlajepa_memv1_video.yaml:56–58`, warm-started from allv2 100K, freeze `vj_encoder,action_model`) → `vlajepa_memv1_stage1` → `vlajepa_memv1_cotrain` → LIBERO eval (`MEMORY_MODE=live`, `NUM_TRIALS=10`).

**Stage 1** (`REPO/scripts/config/vlajepa_memv1_stage1.yaml`): `max_train_steps: 5000` (:127), `robot_only: true` (:129, consumed at train_vlajepa_cotrain.py:108 to skip building the video dataloader; `skip_video_pass` :130 is a **dead knob** — no Python consumer), `robot_world_model_loss: false` (:131, consumed at VLA_JEPA.py:444) → **action loss only; no world-model loss at all in stage 1**. Warm start from the video run's final model (:144) with checkpoint migration allowing missing `memory_module.` / `policy_memory_fusion.` prefixes and rejecting all unexpected keys (:145–151; loader `REPO/starVLA/training/trainer_utils/trainer_tools.py:263–298`; guard `_migration_missing_prefixes` train_vlajepa_cotrain.py:79–89). Freeze `qwen_vl_interface,vj_encoder,vj_predictor` (:161); LRs: base/action_model/memory_module/policy_memory_fusion all 1e-4, qwen 1e-6 (:152–157). So Stage 1's 5K steps trained only: memory module, policy fusion, and the DiT action head, on recurrent robot segments.

**160K decisions**: `decisions_per_step = per_device_batch(1) × 8 GPUs × grad_accum(1) × segment_length(4) = 32` (train_vlajepa_cotrain.py:163–170); 5000 × 32 = 160,000, confirmed by `data/processed_decisions: 160000` in the stage1 wandb summary. Each sample additionally carries ≤8 unsupervised burn-in decisions (not counted).

**Sequence training** (`forward_sequence`, VLA_JEPA.py:428–524): batch of same-episode segments from `sample_segment` (`REPO/starVLA/dataloader/gr00t_lerobot/datasets.py:1717–1794`): J=8 burn-in (left-padded with `None`, replaced by carrier examples with `update_mask=False` at VLA_JEPA.py:451–456, 481–486) + K=4 supervised decisions at raw stride 7 (`segment_length: 4`, `burn_in_max_decisions: 8`, `segment_stride: 7`, `sample_mode: contiguous_segment` — stage1 yaml:103–107). Masks: `loss_mask` true only on the K supervised steps (datasets.py:1765–1766), `update_mask = sequence_valid` (1767), `is_first = base_index==0` (1770). In-model: `reset_mask = is_first & active` (VLA_JEPA.py:476–480); burn-in performs read/write with no loss; **`memory_detach_burn_in: true` detaches state at the burn-in/supervised boundary** (so gradients do not reach burn-in writes) and `memory_bptt_steps: 4` detaches every 4 supervised decisions (VLA_JEPA.py:445–497; config stage1 yaml:141–142). Losses: mean of per-step action losses (:516–518). All rows must be valid at supervised timesteps (:488–491). `forward()` dispatches to `forward_sequence` when examples carry `"steps"` (:419–421). `memory_direct_context_dropout: 0.0` (stage1 yaml:143) is a **dead knob** — never consumed in any `.py`.

**Cotrain** (`REPO/scripts/config/vlajepa_memv1_cotrain.yaml`): 100K steps planned (:126), warm start from stage1 final (:143, migration disabled :144), freeze only `qwen_vl_interface,vj_encoder` (:160) so vj_predictor + action head + memory adapters cotrain; `robot_world_model_loss: true` (:130); per-step VLA sequence pass + independent SSV2 video pass (train_vlajepa_cotrain.py:418–452). Run currently at step 34,729 (`RUNS/vlajepa_memv1_cotrain/summary.jsonl` last line `{"steps": 34729,...}`; `processed_decisions: 1,111,040` = 34,720×32 in last wandb summary). No memv1 rows exist yet under `REPO/results/*` (eval job is `afterok` the full cotrain).

## 10. Memory ↔ world model

- **World-model memory conditioning is disabled by flag**: `framework.memory.world_model_conditioning.enabled: false` (stage1 yaml:69–75, cotrain yaml:69–75). If set true, construction raises `NotImplementedError("world-model memory conditioning is a Phase-3 feature")` (VLA_JEPA.py:139–140). No `world_memory_fusion` module exists anywhere.
- The world loss consumes the raw pre-fusion `qwen.action_tokens` (VLA_JEPA.py:396–398, 284–310); memory reads the same tokens but only ever writes into the **policy** path, so memory cannot affect `wm_loss`. The memory write happens after world/action losses are formed and consumes only current markers, never targets (comment + code VLA_JEPA.py:413–416). Robot-pass `wm_loss` is scaled ×0.1 (:399). Long-term tier similarly hard-blocked: `long_term.enabled: true` → `NotImplementedError` (VLA_JEPA.py:137–138).

## 11. schema_version

`framework.memory.schema_version: 1` (stage1 yaml:58; also present with memory disabled in video yaml:58). Stored as `self.memory_schema_version` (0 when disabled) (VLA_JEPA.py:126); written into every full-state checkpoint sidecar as `memory_schema_version` (train_vlajepa_cotrain.py:282–286; also `REPO/starVLA/training/train_vlajepa_video.py:251`); export hard-fails unless `schema == 1` for memory-enabled runs and cross-checks key prefixes vs `enabled` (export_vlajepa_ckpt.py:39–57).

## 12. Complete memory config knob inventory

Consumed in code:
- `framework.memory.enabled` (VLA_JEPA.py:125), `schema_version` (:126).
- `short_term.enabled` (must be true, :133–134), `num_slots`=8 (:146), `dim`=512 (:143), `num_heads`=8 (:148), `update_gate_init`=0.1 (:149).
- `action_conditioning.enabled` (must be true, :135–136), `zero_init_gate`=false (:151), `gate_init`=1e-3 (:152), `bottleneck_dim`=512 (:156), `num_heads` (:157), `dropout`=0.0 (:158).
- `long_term.enabled` (must be false, :137–138); `world_model_conditioning.enabled` (must be false, :139–140).
- Trainer: `memory_bptt_steps`=4 (VLA_JEPA.py:445), `memory_detach_burn_in`=true (:446), `robot_world_model_loss` (:444), `repeated_diffusion_steps`=4 (:324), `learning_rate.memory_module`/`learning_rate.policy_memory_fusion` (=1e-4; consumed by `build_param_lr_groups`, trainer_tools.py:51–99), `checkpoint_migration.{enabled,strict,allow_missing_prefixes,allow_unexpected_prefixes}` (train_vlajepa_cotrain.py:79–89), `robot_only` (:108).
- Data: `sample_mode: contiguous_segment`, `segment_length: 4`, `burn_in_max_decisions: 8`, `segment_stride: 7` (`REPO/starVLA/dataloader/lerobot_datasets.py:62–97,130–132`; datasets.py:1420–1450), `delete_pause_frame: false` (lerobot_datasets.py:74–77).
- Deployment: env `MEMORY_MODE` ∈ {live, zero} (server_policy.py:44; websocket_policy_server.py:47–49).

Declarative-only (present in YAML, never read by code): `memory.source`, `memory.read_before_write`, `memory.state_dtype` (stage1 yaml:59–61); `world_model_conditioning.{mode,bottleneck_dim,dropout,zero_init_gate,gate_init}` (yaml:70–75); `short_term.update: gated_cross_attention` (yaml:81); `long_term.{type,key_dim,value_dim,retention_half_life}` (yaml:84–88); `trainer.skip_video_pass` (yaml:130); `trainer.memory_direct_context_dropout` (yaml:143); `vla_data.require_same_trajectory`, `emit_episode_metadata` (yaml:107–108).

## 13. Eval-time ablation hooks: EXIST vs DO NOT EXIST

The eval plan (`REPO/docs/VLA-JEPA-Memory-Evaluation-Plan.md:29–44`) requires nine runtime modes; implementation status:

EXIST:
1. `live` — MEMORY_MODE=live (default), full recurrence (websocket_policy_server.py:208–217).
2. `zero` (as implemented) — MEMORY_MODE=zero: fresh learned-init state each call, no writes, fusion still applied (websocket_policy_server.py:210–212). **Does not match the eval-plan `zero` definition** (forced-zero residual).
3. `build_disabled` — config `memory.enabled: false` builds the exact legacy model with no memory keys (VLA_JEPA.py:122–129; export cross-check export_vlajepa_ckpt.py:46–49).
4. Per-episode reset — reset RPC / client / LIBERO interface (websocket_policy_server.py:138–174; model2libero_interface.py:70–76; eval_libero.py:149).
5. Programmatic-only hooks on `predict_action`: arbitrary `memory_state` injection (would enable foreign-state replay), `reset_mask`, `update_memory`, `return_memory_state`, per-session `generator`/`initial_noise` (VLA_JEPA.py:529–538) — none of these are exposed through any server message, env var, or eval flag beyond what MEMORY_MODE maps to.
6. Module-level `bypass=True` kwarg on `ResidualMemoryFusion.forward` (fusion.py:60, 82–83) — exists but is **unwired**: no call site in `VLA_JEPA.py` or serving code; exercised only by `tests/memory/test_memory_fusion.py:65–74`.

DO NOT EXIST (documented in the eval plan but with no implementation anywhere in the repo — verified by grep over `*.py`/`*.sh`):
- `runtime_bypass` (skip read/write/fusion on the trained checkpoint) — no MEMORY_MODE value, no flag.
- True zero-residual mode (eval-plan `zero`).
- `reset_each_decision` mode (the implemented `zero` incidentally behaves as reset-every-decision-without-writes, but no dedicated hook exists).
- `shuffle_within_batch` — nothing.
- `foreign_episode` state replay harness — nothing (no state recording/serialization path either; `MemoryState` is never msgpacked, websocket server keeps it private).
- `short_only` / `long_only` tier ablations — meaningless in Phase 1 (no long tier; `long_term.enabled: true` raises `NotImplementedError`).
- No episode-level machine-readable result records with `memory_mode` fields (eval plan §11); eval output is `eval.log` grep of "Total success rate" (eval_after_run.sbatch:104–107).

## 14. Diagnostics actually implemented

Only four scalars: `memory/working_norm`, `memory/steps`, `memory/active` (recurrent_memory.py:224–228) plus `memory/policy_gate` = `tanh(gate)` (VLA_JEPA.py:391), surfaced via `last_memory_diagnostics` (VLA_JEPA.py:521–524) and logged at train_vlajepa_cotrain.py:462–468. The implementation plan's fuller wishlist (update-gate p05/p95, slot pairwise cosine, residual-injection norm — `REPO/docs/VLA-JEPA-Memory-Implementation-Plan.md:309–316`) is not implemented. Known logging defect: in cotrain (non-robot_only) the video pass nulls the diagnostics before logging, so no `memory/*` metric was recorded for the entire cotrain run (see §5).

## MAP REPORT: training

# VLA-JEPA Training Pipeline Report

Repo root: `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA` (HEAD `25e1882`). All paths below relative to repo root unless absolute.

## 1. Co-training entrypoint and loop

Entrypoint: `starVLA/training/train_vlajepa_cotrain.py`, launched by `cluster/launch_vlajepa_cotrain_8gpu.sbatch:100-112` via `accelerate launch --config_file ./starVLA/config/accelerate/ddp_bf16.yaml` (bf16 mixed precision, MULTI_GPU, 8 procs — `starVLA/config/accelerate/ddp_bf16.yaml:3,8,10`). Plain DDP, no DeepSpeed, `find_unused_parameters=True` (`train_vlajepa_cotrain.py:58`). Model: `VLA_JEPA` framework (`starVLA/model/framework/VLA_JEPA.py:44`).

Per outer step (`VLAMTrainer._train_step`, `train_vlajepa_cotrain.py:418-469`) there are **two full optimizer updates**:
1. **Robot (VLA) pass**: `zero_grad` → `model(batch_vla)` under `torch.autocast("cuda", bfloat16)` → `total_loss = sum(output_dict.values())` → backward → clip → `optimizer.step()` → `lr_scheduler.step()` (lines 422-438). Robot batches are segments (`"steps"` key) so `forward()` dispatches to `forward_sequence` (`VLA_JEPA.py:419-421`).
2. **Video pass**: separate `zero_grad`/forward/backward/clip/step/scheduler-step on the SSV2 batch (lines 440-452), skipped when `trainer.robot_only` (dataloader not built, `train_vlajepa_cotrain.py:108-111`). Config key `skip_video_pass` is **never read by code** (grep: only in yaml).

### Losses — exact formulas
- **Action loss** (flow matching, `starVLA/model/modules/action_model/GR00T_ActionHeader.py:270-317`): `t = (0.999 − Beta(1.5,1.0).sample)/0.999` discretized into 1000 buckets; `noisy = (1−t)·noise + t·actions`; `velocity = actions − noise`; `loss = mean((pred_actions − velocity)²)` (line 316). Target = last `future_action_window_size+1 = 7` action steps (`VLA_JEPA.py:321`); batch repeated `repeated_diffusion_steps=4`× before the head (`VLA_JEPA.py:324-330`); memory fusion applied once before this expansion. Action head runs under explicit bf16 autocast (`VLA_JEPA.py:334-335`).
- **World-model loss** (`VLA_JEPA.py:284-310`): frozen V-JEPA2 teacher `vj_encoder.get_vision_features` under `torch.no_grad` (line 301), two camera views concatenated on channel dim (line 302; effective latent D = 2×encoder hidden); `input_states` = tokens of first `latent_frames−1` frames, `gt_states` = tokens shifted one latent frame (lines 305-306); `predicted_states = vj_predictor(input_states, action_tokens)`; **`world_loss = F.l1_loss(predicted_states, gt_states, reduction="mean")`** (line 308).
- **Weights**: robot-pass wm loss is **hard-coded ×0.1** (`scaled_world_loss = world_loss if not has_actions else world_loss * 0.1`, `VLA_JEPA.py:399`); video-pass wm loss is unscaled (×1.0); action loss ×1.0. The `trainer.loss_scale` yaml (vla 1.0 / vlm 0.1) is **unused** by this trainer — it is only referenced in the legacy `starVLA/training/train_starvla_cotrain.py:387`.
- **Segment unroll** (`VLA_JEPA.py:428-524`): iterates timesteps; burn-in steps run with `include_action_loss=False, include_world_loss=False` (memory write only); supervised steps append `action_loss` and (if `trainer.robot_world_model_loss`, line 444) `wm_loss`; final output = `mean(stack(action_losses))` and `mean(stack(world_losses))` (lines 518-520). Memory BPTT: state detached at entry to the supervised span if any burn-in write happened and `memory_detach_burn_in` (lines 493-494), then detached every `memory_bptt_steps=4` supervised steps (lines 495-497). Memory read happens before losses (`VLA_JEPA.py:388-391`), write only after all loss tensors (`VLA_JEPA.py:415-416`), reads/writes use `qwen.action_tokens` (source `qwen_action_tokens_current_only`); fusion injects into embodied-action tokens.

## 2. Trainable vs frozen parameters (cotrain)

`trainer.freeze_modules: qwen_vl_interface,vj_encoder` (run config line 157). Freezing sets `requires_grad=False` (`starVLA/training/trainer_utils/trainer_tools.py:157-199`); frozen params are also excluded from optimizer param groups (`trainer_tools.py:76-111`). Result:
- **Frozen**: `qwen_vl_interface` (Qwen3-VL-2B incl. resized token embeddings), `vj_encoder` (V-JEPA2 ViT-L teacher; additionally under `no_grad` in forward).
- **Trained**: `vj_predictor` (12-layer AC predictor, falls into "base" LR group @ 3e-5), `action_model` (DiT-B flow head @ 1e-4), `memory_module` (@ 1e-4), `policy_memory_fusion` (@ 1e-4). The configured `qwen_vl_interface: 1e-5` group ends up empty because all its params are frozen.
- Warm start: `trainer.pretrained_checkpoint = .../vlajepa_memv1_stage1/final_model/pytorch_model.pt`, loaded strict full-model (`checkpoint_migration.enabled: false`; `trainer_tools.py:293`).
- Stage boundaries: video stage freezes `vj_encoder,action_model` (`scripts/config/vlajepa_memv1_video.yaml:114`; trains qwen_vl_interface @1e-5 + vj_predictor @3e-5); stage1 freezes `qwen_vl_interface,vj_encoder,vj_predictor` (`scripts/config/vlajepa_memv1_stage1.yaml:161`; trains memory_module/policy_memory_fusion/action_model @1e-4, base 1e-4).

## 3. Optimizer / schedule / clipping

- AdamW, `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=1e-8` (run config lines 167-173; built at `train_vlajepa_cotrain.py:119-125` from `trainer.optimizer.weight_decay` — the separate `trainer.weight_decay: 0.0` key is ignored).
- Per-module LR groups via `build_param_lr_groups` (`trainer_tools.py:51-113`): base 3e-5, action_model 1e-4, memory_module 1e-4, policy_memory_fusion 1e-4 (run config lines 148-153).
- Scheduler: `cosine_with_min_lr`, `min_lr=1e-6` (lines 154-156). Expressed in optimizer-update units: `optimizer_steps_per_training_step=2`, so warmup = 3000×2 = 6000 updates, total = 100000×2 = 200000 updates (`train_vlajepa_cotrain.py:129-145`); scheduler stepped once after each of the two per-step optimizer updates.
- Gradient clipping: `accelerator.clip_grad_norm_(model.parameters(), 1.0)` after each pass (`trainer.gradient_clipping: 1.0`; `train_vlajepa_cotrain.py:431-432, 449-450`). Only the robot-pass norm is logged (`opt/grad_norm_vla`, line 434); the video-pass norm is computed but not logged.
- `gradient_accumulation_steps: 1`; seed `42 + rank` (`train_vlajepa_cotrain.py:194`). `enable_gradient_checkpointing`/`enable_mixed_precision_training` yaml keys are not referenced anywhere in code (grep: no hits).

## 4. Batch composition (robot vs video)

- Robot: `per_device_batch_size: 1` segment × 8 GPUs × accum 1 = **8 segments per outer step** (`_calculate_total_batch_size`, `train_vlajepa_cotrain.py:184-189`). Each segment: 12 slots = up to 8 burn-in decisions (left-padded with `None`) + 4 supervised decisions; `segment_length: 4`, `burn_in_max_decisions: 8`, `segment_stride: 7`, `sample_mode: contiguous_segment`, `require_same_trajectory: true` (run config lines 102-107). Segment construction with `loss_mask` (supervised tail only), `update_mask=sequence_valid`, `is_first` (base_index==0 → memory reset), deterministic per-(epoch,index,seed) RNG: `starVLA/dataloader/gr00t_lerobot/datasets.py:1717-1794`, `1668-1683`. `decisions_per_step = 8×4 = 32` (`train_vlajepa_cotrain.py:164-169`) — confirmed by live sidecar: `processed_decisions 1,111,328 = 34,729×32`. Robot sample dict = `{action, image:[2 view PILs @224], lang, video:[2,8,256,256,3]}` (`datasets.py:1881`); data mix `all_robot`, action_type `delta_qpos`, `with_state: false`.
- Video: 8 SSV2 clips per outer step (per-device 1×8 GPUs), 8 contiguous frames, single view duplicated to 2 views (`starVLA/dataloader/video_datasets.py:62-76`); loss = wm only. Mix ratio robot:video = 1:1 optimizer updates every step.
- Live counters (34,729 steps): `vla_epoch_count: 0`, `vlm_epoch_count: 10`.

## 5. Three-stage pipeline — `cluster/submit_vlajepa_memv1_pipeline.sh`

- Baseline parent: `vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt` (line 10).
- Preconditions: sbatch/squeue exist (35-36); configs+sbatch files non-empty (38-46); baseline ckpt/config/dataset stats exist (47-51); worktree clean and `HEAD == origin/main` (54-61); run dirs must not pre-exist (63-67); no queued jobs with same names (69-73). Manifest `outputs/slurm/vlajepa_memv1-<sha12>-<UTCstamp>.manifest` records config sha256s, job ids, dependencies, status transitions intent→submitting→submitted/failed/interrupted (89-158).
- Chain (all `--export` with `EXPECTED_GIT_SHA`/`EXPECTED_CONFIG_SHA` re-verified inside each sbatch — `launch_vlajepa_cotrain_8gpu.sbatch:20-32`):
  1. **Video pretrain** job `vj-memv1-video`: `launch_vlajepa_video_8gpu.sbatch`, config `scripts/config/vlajepa_memv1_video.yaml`, MICRO=8, `max_train_steps: 50000`, warmup 2000 (no ×2 multiplier in video trainer — `train_vlajepa_video.py:129-135`), trainer `train_vlajepa_video.py` (video wm loss only, memory disabled), warm-started from allv2-100K (yaml line 101). Lines 160-167.
  2. **Memory stage1** job `vj-memv1-stage1` (`afterok:video`): `launch_vlajepa_cotrain_8gpu.sbatch` with `CONFIG=scripts/config/vlajepa_memv1_stage1.yaml`, MICRO=1, 5000 steps, `optimizer_steps_per_training_step: 1`, `robot_only: true`, `robot_world_model_loss: false` (action loss only), warmup 250, parent = `vlajepa_memv1_video/final_model/pytorch_model.pt`, `checkpoint_migration.enabled: true` with `allow_missing_prefixes: [memory_module., policy_memory_fusion.]` (strict-false load rejecting unexpected keys, `trainer_tools.py:274-291`). Lines 169-177.
  3. **Co-train** job `vj-memv1-cotrain` (`afterok:stage1`): same sbatch, `CONFIG=scripts/config/vlajepa_memv1_cotrain.yaml`, MICRO=1, 100000 steps, parent = stage1 final model. Lines 179-187.
  4. **Eval** job `vj-memv1-eval` (`afterok:cotrain`): `eval_after_run.sbatch` with `RUN=<cotrain dir>`, `OUT_PREFIX=VLA-JEPA-memv1-live`, `WITH_STATE=false`, `MEMORY_MODE=live`, `NUM_TRIALS=10`; requires `.training_complete`, exports latest step via `cluster/export_vlajepa_ckpt.py`, runs `examples/LIBERO/eval_libero_vlajepa.sh` over libero_10/goal/object/spatial (`eval_after_run.sbatch:89-110`). Lines 189-195.
- sbatch runtime envelope (both launch scripts): 1 node, 8 GPUs, 96 CPUs, 4h wall, `--signal=B:USR1@180`, `--requeue`; `TRAIN_DEADLINE_EPOCH = now + 4h − 120s` (`launch_vlajepa_cotrain_8gpu.sbatch:78-81`); GPU-util CSV sampler to `outputs/slurm/vlajepa_util_<jobid>.csv` (86-89); requeue via `scontrol requeue` iff incomplete AND (USR1 or exit 0/124/137/143); genuine crashes are not requeued (120-130).
- Pipeline state: video and stage1 have `.training_complete` and `final_model/pytorch_model.pt` (~6.8 GB each); **cotrain is mid-run at step 34729/100000** (no `.training_complete`; checkpoints `step_27752`, `step_30000`, `step_34729`, plus exported `VLA-JEPA-memv1-live-step_34729.pt` and `VLA-JEPA-memv1-zero-step_34729.pt` in `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/checkpoints/`).

## 6. Checkpointing / resume mechanics

- `save_full_state` (`starVLA/training/trainer_utils/run_utils.py:214-246`): collective `accelerator.save_state` into `checkpoints/step_<N>/` (contents observed: `model.safetensors`, `optimizer.bin`, `random_states_{0..7}.pkl`, `custom_checkpoint_0.pkl` = registered lr_scheduler via `register_for_checkpointing`, `train_vlajepa_cotrain.py:220`), rank-0 sidecar `training_state.json` = `{completed_steps, vla_epoch_count, vlm_epoch_count, wandb_run_id, memory_schema_version, processed_decisions}` (`train_vlajepa_cotrain.py:273-291`), `latest.txt` pointer, prune to `keep_last_checkpoints: 3`.
- Triggers: every `ckpt_interval: 10000` steps (line 404) and on graceful stop (deadline − `grace_seconds: 300`, or SIGUSR1/SIGTERM; all-reduced MAX so ranks stop together — `run_utils.py:139-196`; `train_vlajepa_cotrain.py:407-411` then `sys.exit(0)` for requeue).
- Resume (`--trainer.resume_from_latest true` forced by sbatch, line 107): `find_latest_checkpoint` reads `latest.txt` (`run_utils.py:249-272`), `accelerator.load_state`, restores counters (`train_vlajepa_cotrain.py:255-271`); both dataloaders fast-forwarded deterministically: `epoch, offset = divmod(completed_steps, len(dl))`, `set_epoch(epoch)` on loader/sampler/dataset and on the `skip_first_batches` wrapper (`train_vlajepa_cotrain.py:298-341`).
- `eval_interval: 10000` is a barrier-only synchronization no-op — the in-training probe was disabled to keep dataloader positions deterministic (`train_vlajepa_cotrain.py:515-530`).
- Natural completion: final `step_<N>` save, `final_model/pytorch_model.pt` (`accelerator.get_state_dict`), `.training_complete` marker (`train_vlajepa_cotrain.py:551-566`; `run_utils.py:275-283`).

## 7. Logging (everything emitted)

Frequency `logging_frequency: 10`, rank-0 only (`train_vlajepa_cotrain.py:471-488`). TensorBoard tags = wandb keys with `/`→`_` (line 484); writer at `<run>/tensorboard` (line 158).

**Cotrain scalar keys** (39 tags confirmed in live run event files):
- Losses: `loss/vla_action_loss`, `loss/vla_wm_loss`, `loss/vla_total`, `loss/vlm_wm_loss`, `loss/vlm_total` (lines 454-461).
- Opt/data/time: `opt/grad_norm_vla`, `opt/learning_rate`, `opt/epoch`, `data/processed_decisions`, `time/data`, `time/model`, `time/seconds_to_deadline` (lines 392-393, 434, 476-481).
- JEPA scalars every `jepa_log_interval: 50` steps, captured from the **robot** wm pass only (`capture_jepa` toggled off before video pass, lines 423-430; `VLA_JEPA.py:169-187,309`), computed in `starVLA/training/trainer_utils/jepa_analysis.py:79-172`: `jepa/pred_gt_cosine_mean|std|p10`, `jepa/frac_tokens_cos_gt_0.9`, `jepa/pred_gt_l1`, `jepa/pred_gt_l2`, `jepa/pred_feature_std`, `jepa/gt_feature_std`, `jepa/feature_std_ratio`, `jepa/pred_token_variance`, `jepa/gt_token_variance`, `jepa/pred_token_norm`, `jepa/gt_token_norm`, `jepa/pred_participation_ratio`, `jepa/pred_effective_rank`, `jepa/gt_participation_ratio`, `jepa/gt_effective_rank`, `jepa/effective_rank_ratio` (effective rank = exp entropy of singular-value distribution, participation ratio = (Σs²)²/Σs⁴; `jepa_analysis.py:54-74`), `jepa/identity_baseline_cosine`, `jepa/pred_gain_over_identity`, `jepa/input_token_norm`, `jepa/action_token_norm`, `jepa/action_to_state_norm_ratio`, `jepa_view/view{0,1}_cosine`, `jepa_view/view{0,1}_pred_std`.
- JEPA figures every `jepa_figure_interval: 500` under wandb `jepa_media/`: `spectrum` (log SV spectrum), `cosine_hist`, `dim_std`, `pca_scatter` (`jepa_analysis.py:177-278`; `train_vlajepa_cotrain.py:507-509`).
- **Memory diagnostics** — defined as `memory/working_norm`, `memory/steps`, `memory/active` (`recurrent_memory.py:224-228`) and `memory/policy_gate = tanh(fusion.gate)` (`VLA_JEPA.py:391`), logged via `last_memory_diagnostics` (`train_vlajepa_cotrain.py:462-468`). **BUG-grade caveat: they are absent from the live cotrain run** — the video-pass `forward()` sets `last_memory_diagnostics = None` (`VLA_JEPA.py:423-425`) before the trainer reads it after both passes; confirmed empirically: stage1 TB has `memory_active/memory_policy_gate/memory_steps/memory_working_norm`, the cotrain TB does not.
- wandb (`RichWandbLogger`, `run_utils.py:38-135`): project `vla-jepa`, entity `crlc112358`, group `vlajepa-memv1-cotrain`, mode online, run name/id `vlajepa_memv1_cotrain-E20260630-memv1-cotrain`, `resume="allow"` (stable id across requeues — 6 wandb run dirs observed for the same id), tags `["vla-jepa","cotrain"]`, `train/step` defined as the shared x-axis metric; run summary fields `total_params_M`, `trainable_params_M`, `total_batch_size` (`train_vlajepa_cotrain.py:250-253`).
- `summary.jsonl`: **only** `{"steps": <int>, "time": <unix int>}` appended per full-state save (`train_vlajepa_cotrain.py:293-294`). Live file: 8 lines, saves at 6988, 10000, 13911, …, 27752, 30000, 34729 (mix of interval and deadline-stop saves).
- stdout: `accelerator.state` dump (line 59); "***** Cotrain Configuration *****" block — total outer steps, optimizer updates/step (2), total optimizer updates, per-device batch, total batch, supervised decisions/step, resume state, deadline (lines 532-549); LR-group listing (lines 126-128); `🔒 Frozen modules ...`/`📊 model parameter statistics` (`trainer_tools.py:198, 209-214`); `logger.info("Step N | {metrics}")` every 10 steps (line 488); tqdm postfix `data=<s> model=<s>`; `[resume]`, `[data-resume]`, `[stop]`, `[exit]`, `✅ full-state checkpoint @ step N`, `[eval] ... in-training probe disabled` messages.
- Video trainer (`train_vlajepa_video.py`) differences: keys `loss/wm_loss`, `loss/total`, `opt/grad_norm` (not `_vla`), no `data/processed_decisions`, wandb tags `["vla-jepa","video-pretrain"]`, group `vlajepa-memv1-video`; sidecar lacks vlm counters. Stage1 (robot_only) TB tags: `loss_vla_action_loss`, `loss_vla_total`, `memory_*` (4), `opt_*`, `time_*`, `data_processed_decisions`; **no jepa tags** (wm loss disabled ⇒ `_compute_world_loss`/capture never runs).

## 8. Live run memory + loss config (`/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/config.yaml`)

Memory (lines 55-87): `enabled: true`, `schema_version: 1`, `source: qwen_action_tokens_current_only`, `read_before_write: true`, `state_dtype: float32`; `action_conditioning`: `mode: residual_cross_attention`, `bottleneck_dim: 512`, `dropout: 0.0`, `zero_init_gate: false`, `gate_init: 0.001`; `world_model_conditioning.enabled: false` (Phase-3, `NotImplementedError` if enabled — `VLA_JEPA.py:139-140`); `short_term`: 8 slots × 512 dim, 8 heads, `update: gated_cross_attention`, `update_gate_init: 0.1`; `long_term.enabled: false` (gated_delta stub, Phase-2, `VLA_JEPA.py:137-138`). Module math: FP32-forced gated slot update `new = (1−σ(W_g[LN(prev);ctx]))·prev + σ(...)·tanh(W_c·ctx)` with ctx = MHA(slots+slot_ids → projected source tokens), gate bias init `logit(0.1)` (`recurrent_memory.py:252-273, 71-73`); fusion `out = consumer + tanh(scalar gate)·W_out(MHA(query=proj(LN(consumer)), kv=proj(LN(memory))))`, gate scalar init 0.001 (`fusion.py:42, 86-100`).

Trainer/loss (lines 122-177): `max_train_steps: 100000`, `optimizer_steps_per_training_step: 2`, `robot_only: false`, `skip_video_pass: false` (unused key), `robot_world_model_loss: true`, `num_warmup_steps: 3000`, `save_interval/ckpt_interval: 10000`, `keep_last_checkpoints: 3`, `eval_interval: 10000`, `resume_from_latest: true`, `grace_seconds: 300`, `jepa_log_interval: 50`, `jepa_figure_interval: 500`, `memory_bptt_steps: 4`, `memory_detach_burn_in: true`, `memory_direct_context_dropout: 0.0` (**no code references — unused**), `pretrained_checkpoint: .../vlajepa_memv1_stage1/final_model/pytorch_model.pt`, `checkpoint_migration.enabled: false`, LRs `{base: 3e-5, qwen_vl_interface: 1e-5, action_model: 1e-4, memory_module: 1e-4, policy_memory_fusion: 1e-4}`, `cosine_with_min_lr` + `min_lr: 1e-6`, `freeze_modules: qwen_vl_interface,vj_encoder`, `loss_scale: {vla: 1.0, vlm: 0.1}` (unused; robot wm ×0.1 is hardcoded at `VLA_JEPA.py:399`), `max_grad_norm: 1.0` (unused; clipping uses `gradient_clipping: 1.0`), `logging_frequency: 10`, `gradient_accumulation_steps: 1`, AdamW `{betas: [0.9,0.95], eps: 1e-8, weight_decay: 1e-8}`. Framework: Qwen3-VL-2B (hidden 2048), DiT-B action head (16 layers, action_dim 7, horizon 7, 4 inference timesteps), V-JEPA2 vitl-fpc64-256 teacher + 12-layer/8-head predictor, 8 frames, 8 action tokens/timestep, 32 embodied-action tokens. Live config differs from repo `scripts/config/vlajepa_memv1_cotrain.yaml` only by CLI override `resume_from_latest: true` and yaml formatting.

## MAP REPORT: datasets

# VLA-JEPA Datasets Report

## 1. Robot data on disk (LeRobot format)

Root: `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot` (verified with `ls`; 7 dataset dirs present). Canonical list: `cluster/prepare_all_robot_data.py:47-55` (`DATASETS`), consumed as mixture `all_robot` in `starVLA/dataloader/gr00t_lerobot/mixtures.py:56-64`.

Per-dataset facts (from each `meta/info.json` on disk + episode-length scan of `meta/episodes.jsonl` + byte counts in `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/all_robot_pipeline/datasets_complete.json`, completed 2026-06-28):

| dataset | episodes | frames | tasks | videos | fps / robot | ep len min/mean/median/p90/max | eps ≥85 frames | parquet / video bytes |
|---|---|---|---|---|---|---|---|---|
| libero_object_no_noops_1.0.0_lerobot | 454 | 66,984 | 10 | 908 (2 views) | 20 / franka | 114/147.5/146/168/254 | 100% | 7.1 MB / 533 MB |
| libero_goal_no_noops_1.0.0_lerobot | 428 | 52,042 | 10 | 856 | 20 / franka | 75/121.6/105/184/270 | 95.3% | 5.9 MB / 331 MB |
| libero_spatial_no_noops_1.0.0_lerobot | 432 | 52,970 | 10 | 864 | 20 / franka | 75/122.6/123/149/193 | 98.8% | 6.0 MB / 363 MB |
| libero_10_no_noops_1.0.0_lerobot | 379 | 101,469 | 10 | 758 | 20 / franka | 150/267.7/259/342/505 | 100% | 9.3 MB / 626 MB |
| droid_lerobot | 92,233 | 27,044,326 | 31,308 | 276,699 (3 views) | 15 / franka | 1/293.2/220/547/4498 | 95.8% (88,398) | 2.56 GB / 389.5 GB |
| bridge_orig_1.0.0_lerobot | 53,192 | 1,893,026 | 19,974 | 212,768 (4 views) | 5 / widowx | 1/35.6/35/47/117 | 1.2% (641) | 346 MB / 21.3 GB |
| fractal20220817_data_0.1.0_lerobot | 87,212 | 3,786,400 | 599 | 87,212 (1 view) | 3 / google_robot | 2/43.4/40/68/650 | 3.5% (3,052) | 630 MB / 21.3 GB |

- LIBERO suites present: object, goal, spatial, 10 ("no_noops" conversions; local, no pinned HF revision — `prepare_all_robot_data.py:490-494` notes local LIBERO conversions predate `stats.json`). `libero_90` is NOT on disk (commented out, `mixtures.py:20,44`).
- DROID/Bridge/Fractal are HF snapshots pinned by revision (`prepare_all_robot_data.py:57-63`): `IPEC-COMMUNITY/droid_lerobot@96dd57f36d85...`, `bridge_orig_lerobot@0e9d76d07e9d...`, `fractal20220817_data_lerobot@91bf7d7f7ce5...` (revisions in `all_robot_pipeline/datasets_complete.json`).
- `cluster/dl_droid.sbatch`: self-requeueing 4h batch job that runs `vlajepa_stage/dl_droid.sh` with a GPU-keepalive matmul loop (idle-GPU watchdog dodge); declares DROID done at ≥275,000/276,699 mp4s.
- Long-episode suitability for memory: **DROID** (median 220 frames @15 fps ≈ 14.7 s, max 4,498 ≈ 5 min) and **libero_10** (mean 268 @20 fps) are the only corpora where nearly all episodes fill the full 85-frame segment window; Bridge/Fractal episodes are mostly too short for any burn-in history.

### DROID caches (`cluster/rebuild_droid_cache.py`)
- Validates all 92,233 parquets (27,044,326 frames), schema `observation.state`(8)+`action`(7)+scalars (`rebuild_droid_cache.py:549-558`), then atomically publishes `meta/stats_gr00t.json` (full-data mean/std/min/max/q01/q99 for state[8], action[7]) and pause-filtered step cache `meta/steps_332420bad1ab.pkl` (`PREFERRED_STEPS_NAME`, `rebuild_droid_cache.py:39`; legacy `steps_2d5a34b904d2.pkl:40`).
- Pause filtering: keep step if |Δtranslation|>5e-4 or gripper change (`rebuild_droid_cache.py:37,328-332`; mirrors `datasets.py:69,536-549`).
- Marker `/lustre/.../vlajepa_stage/all_robot_pipeline/droid_cache_complete.json`: `total_steps=23,835,621`, `unique_trajectory_ids=91,301` of 92,233, steps pkl 190,685,387 bytes, completed 2026-06-28. Both cache files verified present in `droid_lerobot/meta/` (`ls` confirmed).

## 2. Video pretraining corpus

- Something-Something v2 at `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/ssv2/20bn-something-something-v2`: **220,847 .webm files** (verified by `ls | wc -l`; 19.48 GB per audit), labels `ssv2/labels_all.csv`: **193,690** headerless `id;text` lines (verified `wc -l`). Expected counts hard-asserted in configs and `prepare_all_robot_data.py:44-45`. `ssv2_raw/` holds download docs + `labels.json` + `test-answers.csv`.
- Loader `starVLA/dataloader/video_datasets.py` (`VideoFolderDataset`): samples `n_frames = cfg.framework.vj2_model.num_frames = 8` **contiguous** frames from a random start (`video_datasets.py:150-156`), resizes to 256×256 (`crop_h/w = video_resolution_size`), collate duplicates into 2 views → `video [2,8,256,256,3]`, `image` = frame 0 at 224² (`video_datasets.py:62-76`). Videos with no label (the ~27k unlabeled test clips) get fallback caption "Completing something that humans might want to do." (`video_datasets.py:130-133`). DataLoader: DistributedSampler shuffle, bs 8/GPU, 16 workers (`starVLA/dataloader/__init__.py:107-137`). 10 retries with random re-index on decode failure (`video_datasets.py:180-187`).

## 3. Robot dataloader pipeline (windowing / history — CRITICAL)

Entry: `starVLA/dataloader/__init__.py:38-70` → `get_vla_dataset(action_horizon=cfg.framework.action_model.action_horizon=7, video_horizon=cfg.framework.vj2_model.num_frames=8, ...)`; DataLoader num_workers=8, identity `collate_fn` (`lerobot_datasets.py:10-11`).

**Per-decision window** (`lerobot_datasets.py:31-35` builds each robot-type config with `observation_indices=list(range(8))`, `action_indices=list(range(7))`; classes in `gr00t_lerobot/data_config.py:457-534,148-378`):
- Video delta indices **[0..7]**: 8 frames from the decision index t **forward** (t..t+7, edge-clamped `datasets.py:940-941`). **There are no negative delta indices anywhere — a single decision contains ZERO past frames.**
- Action delta indices **[0..6]**: 7-step future action chunk, zero-padded past episode end (`datasets.py:1010-1016`).
- `with_state: false` in all cluster configs (heterogeneous embodiments).
- `_format_step` output per decision (`datasets.py:1845-1888`): `video [2 views, 8, 256, 256, 3]` (LIBERO=primary+wrist; DROID uses views 0 and 2 of 3; single-view datasets duplicate), `image` = frame-0 224² per view, `action` float16 [7,7], `lang`.

**single_step mode** (baseline `vlajepa_cotrain_all.yaml`, no `sample_mode` key → default): 1 decision per sample, drawn from the pause-filtered `all_steps` cache (`datasets.py:425-495`; `delete_pause_frame` defaults **True** in `get_vla_dataset`, `lerobot_datasets.py:74-78`). So the baseline model sees 8 current+future frames and no history.

**contiguous_segment mode** (memv1 stage1 + cotrain; `scripts/config/vlajepa_memv1_{stage1,cotrain}.yaml`: `sample_mode: contiguous_segment`, `segment_length: 4`, `burn_in_max_decisions: 8`, `segment_stride: 7`, `delete_pause_frame: false`, `per_device_batch_size: 1`):
- One sample = **12 decision slots: 8 burn-in (left-padded with `None` when the episode is short) + 4 supervised decisions**, on an episode-anchored lattice `0, 7, 14, ...` (`datasets.py:1568-1627,1717-1794`). Stride 7 = action_horizon → consecutive non-overlapping action chunks.
- Valid supervised start requires `L ≥ 29` frames (`last_start = L−1−max_delta(7)−supervised_span(21)`, `datasets.py:1605-1614`); a **full** 8-decision burn-in requires `L ≥ 85` frames (hence the ≥85 column above).
- **Temporal bound per training sample: at most 85 consecutive frames** — up to 56 frames (8 decisions × stride 7) of past context before the supervised window plus 29 frames (4×7 stride + 8-frame video window). At 15 fps DROID ≈ 5.7 s total, ≈ 3.7 s of "past"; at 20 fps LIBERO ≈ 4.25 s.
- Segment dict fields: `steps, dataset_id, episode_id, base_indices, segment_start, is_first, is_last, sequence_valid, loss_mask` (True only on final 4), `update_mask` (`datasets.py:1783-1794`). Deterministic per-(epoch,index,seed) RNG (`datasets.py:1668-1683`); corrupt-video decode retried with up to 10 deterministic alternate segments (`datasets.py:76,1899-1918`).
- Model side (`starVLA/model/framework/VLA_JEPA.py:428-517`): `forward_sequence` unrolls the 12 slots; `memory_bptt_steps: 4` and `memory_detach_burn_in: true` (both memv1 yamls) → **gradient horizon = the 4 supervised decisions (28 frames); burn-in influences memory content only through detached writes**. So memory can *condition* on ≤56 frames of past but can only *learn credit assignment* across 4 decisions.
- Mixture sampling: weights all 1.0 and `balance_dataset_weights`/`balance_trajectory_weights` default **False** (`lerobot_datasets.py:56-57`, not overridden in `__init__.py:44-57`) → each of the 7 datasets is drawn **uniformly at 1/7 probability** (`datasets.py:1504-1523,1694-1696`), heavily oversampling LIBERO relative to DROID per-frame. Epoch length = max(dataset_len/weight) over weight-1.0 datasets (`datasets.py:1947-2010`).

## 4. dataset_statistics.json

Written once by rank 0 at run start: `vla_dataset.save_dataset_statistics(output_dir/"dataset_statistics.json")` (`starVLA/dataloader/__init__.py:66-69`; implementation `datasets.py:2211-2312`). Structure: one entry per embodiment tag (`franka` = 4 LIBERO, `oxe_droid`, `oxe_bridge`, `oxe_rt1`; mapping `gr00t_lerobot/embodiment_tags.py:68-72`, note `droid_libero → OXE_DROID`), each with `action{mean,std,max,min,q01,q99,mask[7]}` (mask False on gripper dim), `state{...}[8]`, `num_transitions`, `num_trajectories`. Per-tag stats are merged from `meta/stats_gr00t.json` per dataset with `percentile_mixing_method: "min_max"` (`datasets.py:1422-1424,2104-2106`).

On-disk values:
- `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/dataset_statistics.json` (contiguous mode → transitions = all frames): franka 273,465/1,693; oxe_droid 27,044,326/92,233; oxe_bridge 1,893,026/53,192; oxe_rt1 3,786,400/87,212.
- `.../vlajepa_cotrain_allv2/dataset_statistics.json` (single_step + pause filter): franka 272,104; oxe_droid 23,835,621; oxe_bridge 1,863,900; oxe_rt1 3,449,894.
- Also present in `vlajepa_memv1_stage1`, `vlajepa_memv1_smoke_109439b`, `vlajepa_cotrain`, `vlajepa_cotrain_all` run dirs. Used at eval time as the action un-normalization key.

## 5. Held-out / validation splits

**There is no held-out split of any training corpus.** All robot datasets load with `mode="train"` (`lerobot_datasets.py:56`, `datasets.py:1431` — mode only fixes RNG); `load_all_data_for_training: true` in every config. The in-training eval probe is deliberately disabled to preserve dataloader determinism (`train_vlajepa_cotrain.py` `_safe_eval`: "in-training probe disabled to preserve deterministic dataloader position"). Evaluation = LIBERO **simulator** rollouts as a dependent Slurm job: `cluster/eval_libero_vlajepa.sbatch` → `examples/LIBERO/eval_libero_vlajepa.sh:53` suites `libero_10 libero_goal libero_object libero_spatial`, `--args.num-trials-per-task`. SSV2 `test-answers.csv` exists but is unused; unlabeled SSV2 test clips are *included in training* with a fallback caption.

## 6. Training runs on disk (`/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs`)

- `vlajepa_memv1_video`: complete at 50k steps (checkpoints `step_49500`, `step_50000`, `final_model`); SSV2-only trainer `train_vlajepa_video.py` (`launch_vlajepa_video_8gpu.sbatch:34`), memory disabled, warm-started from `vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt`.
- `vlajepa_memv1_stage1`: complete at 5k steps (`step_5000`, `final_model`); `robot_only: true`, `skip_video_pass: true`, contiguous_segment, memory enabled, from memv1_video final.
- `vlajepa_memv1_cotrain`: **in progress at step 34,729 / 100,000** (checkpoints `step_27752/30000/34729`, live export `VLA-JEPA-memv1-live-step_34729.pt` 7.47 GB, mtime Jul 2); contiguous_segment + SSV2 video pass, `optimizer_steps_per_training_step: 2`, Qwen+VJ-encoder frozen.
- Others: `vlajepa_cotrain_allv2` (100k baseline parent), `vlajepa_cotrain_all`, `vlajepa_cotrain`, `vlajepa_memv1_smoke_109439b`, `vlajepa_resume_smoke`, `vlajepa_smoke`, `vlajepa_video`. Pipeline chain submitted by `cluster/submit_vlajepa_memv1_pipeline.sh` (video → stage1 → cotrain → eval, clean-worktree + origin/main SHA enforced).

## 7. Other data on disk (NOT wired into VLA-JEPA)

`/lustre/fsw/portfolios/edgeai/users/chrislin/datasets/`: `agibotworld2026` (167 GB; temporal-grounding build with `build_summary.json`, `source_archives`), `FoxBrain` (278 GB), `RH20T-P` (download/extract scripts). No VLA-JEPA code references these (repo grep matched only the GR00T tag enum `embodiment_tags.py`). `anamnesis_stage/LIBERO` is the LIBERO benchmark checkout used by the eval env. Model weights referenced by all configs: `vlajepa_stage/models/Qwen3-VL-2B-Instruct` and `vlajepa_stage/models/vjepa2-vitl-fpc64-256` (both on disk).

## MAP REPORT: evaluation

# VLA-JEPA Evaluation Infrastructure Report

All paths relative to `REPO=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA` unless absolute.

## 1. Overall LIBERO eval flow

Two-process architecture per suite: a GPU model server (`deployment/model_server/server_policy.py`, conda env `VLA_JEPA`) and a MuJoCo simulator client (`examples/LIBERO/eval_libero.py`, conda env `vlajepa_eval`), connected via WebSocket+msgpack. Orchestrated by `examples/LIBERO/eval_libero_vlajepa.sh`, which launches 4 suites in parallel, one server+sim pair per GPU (`eval_libero_vlajepa.sh:66-82`): server on GPU `index` and port `base_port+index` (base 15083, `eval_libero_vlajepa.sh:54`), 30s sleep for the 7.5GB model to load (`:74`), sim launched against the same port. Server stdout→`results/<suite>/<ckpt>/server.log` (`:72`), sim stdout→`results/<suite>/<ckpt>/eval.log` (`:80`). Checkpoints must first be exported via `cluster/export_vlajepa_ckpt.py <run_dir> <step_dir> <name>.pt` (`eval_libero_vlajepa.sh:5-7`). Env plumbing: `MUJOCO_GL=egl` with a custom NVIDIA EGL ICD (`eval_libero_vlajepa.sh:12-24`), `LIBERO_HOME=/lustre/fsw/portfolios/edgeai/users/chrislin/anamnesis_stage/LIBERO` (`:28`), `LIBERO_CONFIG_PATH=$HOME/.libero` seeded if absent (`:32-42`).

An older environment-specific variant `examples/LIBERO/eval_libero.sh` exists (hardcoded `/home/dataset-local/LIBERO` paths, 50 trials/task, `eval_libero.sh:4,20`); it is not cluster-portable.

## 2. Client-server protocol

- Transport: `websockets` + msgpack-numpy. Client: `deployment/model_server/tools/websocket_policy_client.py`; server: `deployment/model_server/tools/websocket_policy_server.py`. Server max message 256MB (`websocket_policy_server.py:61`), client `max_size=None` (`websocket_policy_client.py:49`). Server sends metadata dict `{"env": "simpler_env"}` on connect (`server_policy.py:43`, `websocket_policy_server.py:71`).
- Message envelope: `{"type": "ping|reset|infer", "request_id": hex, "payload": {...}}`; flat dicts treated as payload; legacy flat `{"reset": True, "instruction": ...}` converted to reset (`websocket_policy_server.py:97-133`). Responses always `{"status","ok","type","request_id", data|error}` (`:103-111`); `_route_message` never raises (`:111`).
- `reset`: payload `{episode_id, episode_seed}`; server creates a fresh `torch.Generator` on the policy device seeded with `episode_seed`, then transactionally commits `generator`, `memory_state=None`, `episode_id`, `batch_size=None`, `ready=True` (`websocket_policy_server.py:147-166`). Failed reset (e.g. seed `2**100`) leaves prior session state intact (tested in `tests/memory/test_websocket_memory.py:124-140`).
- `infer`: payload carries `batch_images`, `instructions`, `unnorm_key`, `do_sample`, `use_ddim`, `num_ddim_steps`, optional `state` (built at `examples/LIBERO/model2libero_interface.py:111-121`). Server converts images to PIL via `image_tools.to_pil_preserve` (`websocket_policy_server.py:201`, `tools/image_tools.py:61-76`) and calls `policy.predict_action(**payload)`.

## 3. Server-side memory state: per-step hold, per-episode reset

Per-connection state lives in `_ConnectionSession` (dataclass: `memory_state`, `generator`, `episode_id`, `batch_size`, `ready`; `websocket_policy_server.py:19-27`), instantiated once per WebSocket handler (`:69`) — so state is isolated per connection and dropped on disconnect (`:78-79`).

Infer path when `policy.memory_enabled` (`websocket_policy_server.py:188-217`):
- Requires explicit prior reset (`:189-190` raises `"memory-enabled inference requires an explicit reset"`).
- Enforces exactly one episode row per connection (`:192-193`, "Phase-1 serving") and constant batch size within a session (`:195-198`).
- Injects `memory_state=None if memory_mode=="zero" else state_before`, `return_memory_state=True`, `update_memory=(memory_mode=="live")`, `generator=session.generator` (`:209-214`).
- Commits `session.memory_state = candidate_state` only in `live` mode (`:216-217`) — so state carries across steps within an episode, and `reset` (called by the sim client at every episode start) clears it.
- Transactional failure rollback: RNG state snapshot `rng_before` (`:202-206`) and `state_before` (`:207`) restored on any exception (`:222-226`); tested at `test_websocket_memory.py:98-114`.
- Stateless (non-memory) policies skip all of this; only the generator is forwarded if present (`:219-221`), and batch size may change (`test_websocket_memory.py:149-164`).

Episode/task reset on the sim side: `eval_libero.py:149` calls `model.reset(task_description=...)` before every episode (before `env.reset()` at `:150`). `M1Inference.reset` (`model2libero_interface.py:70-87`) increments `_episode_counter` and sends `client.reset(instruction, episode_id=f"libero-{n}", episode_seed=n)` (`:72-76`), plus clears client-side ensembler/image history/sticky-gripper state. The client also auto-resets if `task_description` changes mid-run (`model2libero_interface.py:104-106`). A new task suite does not get a new connection per task — one `M1Inference`/connection per suite for all 10 tasks × trials (`eval_libero.py:122-128` constructed once before the task loop).

In `predict_action` (`starVLA/model/framework/VLA_JEPA.py:527-599`): if memory enabled and `memory_state is None`, a fresh `init_state` is created (`:558-559`); `reset_state` applied with `reset_mask` (default all-False; `:560-565`); `memory_module.read` produces tokens fused into embodied tokens via `policy_memory_fusion` (`:566-568`); after action prediction, `memory_module.write` runs only if `update_memory=True` (`:584-585`); returned state is detached (`:586-587`). Server keeps memory/fusion math FP32 even under `--use_bf16` (`server_policy.py:28-31`).

## 4. memory_mode plumbing

- Server reads `MEMORY_MODE` env var, default `"live"` (`server_policy.py:44`); `WebsocketPolicyServer.__init__` validates against `{"live","zero"}` (`websocket_policy_server.py:47-49`).
- `zero` mode: every infer gets `memory_state=None` and `update_memory=False`; `session.memory_state` never committed (`websocket_policy_server.py:209-217`; tested `test_websocket_memory.py:116-122`).
- `eval_libero_vlajepa.sh` does NOT set or forward `MEMORY_MODE` explicitly — the server subprocess inherits it from the environment (`eval_libero_vlajepa.sh:70-72`). `cluster/eval_after_run.sbatch` validates and exports it (`eval_after_run.sbatch:34,43-46,99`), logs it (`:100,104,109`). The memv1 pipeline sets `MEMORY_MODE=live` on the eval job (`cluster/submit_vlajepa_memv1_pipeline.sh:193`) and records `eval_memory_mode=live` in the manifest (`:127`).
- Modes required by the plan but NOT implemented in serving: `runtime_bypass`, `reset_each_decision`, `foreign_episode`, `shuffle_within_batch`, `short_only`, `long_only` (`docs/VLA-JEPA-Memory-Evaluation-Plan.md:28-44`). Only `live`/`zero` exist.

## 5. Client-side inference details (M1Inference)

- Action unnormalization happens client-side using per-checkpoint norm stats read from the exported `.pt` via `read_mode_config` (`model2libero_interface.py:62,156-165`); `unnorm_key` selects the stats block (`:164`, `_check_unnorm_key` `:210-228`).
- Action chunking: `action_chunk_size = future_action_window_size + 1` from the checkpoint config (`model2libero_interface.py:167-171`); all configs use `future_action_window_size: 6` (e.g. `scripts/config/vlajepa_memv1_cotrain.yaml:28`) → chunk 7. The client calls the server only when `step % 7 == 0` and replays the cached chunk otherwise (`model2libero_interface.py:123-132`) — i.e., the memory writes once per 7 env steps.
- Unnormalize: clip to [-1,1], binarize gripper dim at 0.5, min/max scale (`model2libero_interface.py:142-154`).
- Inference flags sent per request: `do_sample=False`, `use_ddim=True`, `num_ddim_steps=10` (`model2libero_interface.py:111-118,28-29`).
- Dead code: `AdaptiveEnsembler` is constructed (`:55-56`) and reset (`:79-80`) but never used in `step()`; `image_history` (`horizon=0`) likewise unused.

## 6. eval_libero.py loop, suites, flags

- Args dataclass (`eval_libero.py:49-87`): `host=127.0.0.1`, `port=10093`, `resize_size=[224,224]`, `task_suite_name="libero_goal"` default, `num_steps_wait=10`, `num_trials_per_task=50` default (overridden to 10 by the shell: `eval_libero_vlajepa.sh:56` `NUM_TRIALS:-10`), `seed=7`, `with_state="true"` (string compare at `:210`), `unnorm_key="franka"` (comment: multi-embodiment checkpoints have one stats block per robot tag; LIBERO must select Franka, `:83-85`).
- Suites and horizons (`eval_libero.py:109-120`): `libero_spatial` max_steps=250 (longest demo 193), `libero_object` 280 (254), `libero_goal` 300 (270), `libero_10`/`libero_mix` 520 (505), `libero_90` 400 (373). Each of the 4 evaluated suites has 10 tasks (`task_suite.n_tasks`, `eval_libero.py:102`; confirmed 10 per-task "Current task success rate" lines and "Total episodes: 100" in `results/libero_10/VLA-JEPA-allv2-step_100000/eval.log`). Standard LIBERO semantics: spatial=layout/spatial-relation variation, object=object variation, goal=goal/procedure variation, libero_10=10 long-horizon tasks. `libero_mix` takes `category_value` (Background Textures / Camera Viewpoints / Language Instructions / Light Conditions / Objects Layout / Robot Initial States / Sensor Noise, `eval_libero.py:61-68,98-99`) — a LIBERO-Plus-style perturbation suite.
- Episode loop: fixed init states `task_suite.get_task_init_states(task_id)[episode_idx]` (`:138,153`); first 10 steps are dummy actions `[0]*6+[-1]` to let objects settle (`:34,166-173`); images rotated 180° to match training preprocessing (`:175-179`); state = `eef_pos ⊕ quat2axisangle(eef_quat) ⊕ gripper_qpos` (`:184-190`), sent only if `with_state=="true"` (`:210-211`); gripper binarized to ±1 via `>0.5` (`:36-40,226`); episode success = env `done` (`:244-247`).
- torch.load compat shim forcing `weights_only=False` for LIBERO init-state pickles (`eval_libero.py:18-26`).

## 7. Seeds / determinism

- `np.random.seed(args.seed=7)` (`eval_libero.py:94`); `env.seed(args.seed)` per task with note that seed affects object positions even with fixed init state (`:303-305`).
- Per-episode policy RNG: `episode_seed = client episode counter` (1-based, monotonically increasing across episodes and tasks within one suite run; `model2libero_interface.py:71-76`) → server per-session `torch.Generator.manual_seed` (`websocket_policy_server.py:151-153`) → passed to the flow-matching head as `generator` (`VLA_JEPA.py:576-580`). So determinism holds only for the identical serial episode order; it does not implement the plan's `(eval_seed, suite, task, initial_state_id, decision_index)` noise derivation (`docs/VLA-JEPA-Memory-Evaluation-Plan.md:121`).
- `predict_action` also accepts `initial_noise` (`VLA_JEPA.py:536`) — unused by the serving path (hook for pre-sampled-noise paired evals).

## 8. What is logged vs discarded per episode/step

Logged:
- Per-episode replay video: agentview frames only, preprocessed/rotated, mp4 @10fps, name `rollout_{task_desc_underscored_truncated80}_{md5-8}_episode{i}_{success|failure}.mp4` (`eval_libero.py:44-47,254-262`).
- Success: only as `logging.info` lines in `eval.log` — per-episode `Success: {done}` + running totals (`:269-273`), per-task `Current task success rate` (`:276-278`), final `Total success rate` / `Total episodes` (`:283-286`). No JSON/CSV episode record exists, despite the machine-readable per-episode artifact required by `docs/VLA-JEPA-Memory-Evaluation-Plan.md:307-334` ("Aggregate tables must be generated from episode records"). Downstream scripts grep eval.log for `Total success rate` (`cluster/eval_after_run.sbatch:104-108`).

Discarded:
- Actions: `full_actions` collected and stacked (`eval_libero.py:159,239,264`) but `np.save` is commented out (`:265`) — per-step actions are thrown away.
- Wrist images: never saved (only appended to the observation, `:177-179,196-198`).
- Per-step latency: measured (`:213-217`) but the print is commented (`:218`).
- `embodied_action_tokens`: returned by the server in every infer response (`VLA_JEPA.py:591-596`) but the client uses only `normalized_actions` (`model2libero_interface.py:128-130`).
- env `reward`/`info` ignored (`:243`).

Rerun hazard: `eval_libero_vlajepa.sh` reuses `results/<suite>/<ckpt>` across runs — `eval.log` is overwritten (`>` redirect, `:80`) and old videos remain, so e.g. `results/libero_10/VLA-JEPA-allv2-step_100000/` contains both `..._episode1_success.mp4` and `..._episode1_failure.mp4` from different runs, and its current eval.log shows `Total success rate: 0.72` while `docs/VLA-JEPA-Memory-Evaluation-Plan.md:9-19` records the canonical allv2 baseline as 78/92/98/96 (overall 91%, 10 trials/task, `WITH_STATE=false`). Also a junk directory literally named `results/libero_10 libero_goal libero_object libero_spatial/` exists (unquoted-TASKS artifact from an old run) containing `VLA-JEPA-cotrain-step50000`.

## 9. Per-step memory internals: computed but NOT logged

- `memory_module.read` computes diagnostics `{working_norm, steps, active}` per call (`starVLA/model/modules/memory/recurrent_memory.py:224-229`); `predict_action` stores them on `self.last_memory_diagnostics` (`VLA_JEPA.py:566-567,588-590`) but excludes them from `public_output` (`:591-596`). The websocket server returns only `output_dict` (`websocket_policy_server.py:215,240-247`) and never reads `policy.last_memory_diagnostics` — so at serve time these are computed every step and dropped. This is the natural hook for per-step memory logging (add diagnostics to `public_output` or have the server attach `self._policy.last_memory_diagnostics` to `data`).
- `session.memory_state` (FP32 working slots + steps/valid, `starVLA/model/modules/memory/state.py`) is held in RAM only; never serialized/hashed/dumped, so the plan's state-norm/foreign-state-replay ablations (`Evaluation-Plan.md:44,184,192,213-219`) have no capture mechanism yet.
- Injected-residual norm and fusion-gate value (`Evaluation-Plan.md:192,217`) are not in `read` diagnostics at all (`starVLA/model/modules/memory/fusion.py` emits nothing at inference).
- During training the same diagnostics ARE logged as `memory/*` scalars to TB/wandb (`starVLA/training/train_vlajepa_cotrain.py:462-468`).

## 10. Batch/cluster wrappers and runs

- `cluster/eval_libero_vlajepa.sbatch`: generic; pass `CKPT`/`NUM_TRIALS` via `--export`; 1 node, 8 GPUs, 4h (`:1-24`). Currently pending jobs `vj-m1-ev34k-live` (6040994) and `vj-m1-ev34k-zero` (6040995) use this script — live vs zero memory eval of memv1 step_34729; exports `VLA-JEPA-memv1-live-step_34729.pt` and `VLA-JEPA-memv1-zero-step_34729.pt` already exist in `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/checkpoints/`, but no `results/*/VLA-JEPA-memv1-*` directories exist yet.
- `cluster/eval_after_run.sbatch`: dependent eval; validates `RUN`, `OUT_PREFIX`, `MEMORY_MODE∈{live,zero}`, `WITH_STATE∈{true,false}`, `NUM_TRIALS` (`:31-54`); requires `.training_complete` marker (`:58-61`, currently absent from `vlajepa_memv1_cotrain/` — cotrain job 6029057 still RUNNING, eval 6029058 pending `afterok`), config + `dataset_statistics.json` (`:62-66`), `checkpoints/latest.txt` → `step_N` → `model.safetensors` (`:68-78`); exports checkpoint if missing (`:89-97`); runs `eval_libero_vlajepa.sh` (`:101`); prints per-suite `Total success rate` summary (`:104-109`).
- `cluster/eval_after_100k.sbatch` (cotrain run, `WITH_STATE=true`, NUM_TRIALS=10) and `cluster/eval_after_all.sbatch` (allv2 run, `WITH_STATE=false` because "all_robot trains stateless"; NUM_TRIALS=10).
- `cluster/submit_vlajepa_memv1_pipeline.sh`: one-shot video→stage1→cotrain→eval chain with `afterok` deps (`:160-195`); eval env: `OUT_PREFIX=VLA-JEPA-memv1-live, WITH_STATE=false, MEMORY_MODE=live, NUM_TRIALS=10` (`:193`); manifest at `outputs/slurm/vlajepa_memv1-109439b53008-20260701T072000Z.manifest` (git sha, config sha256s, job ids 6018626-29).

## 11. Offline (non-simulator) evaluation

No standalone offline policy-eval script (no open-loop action-MSE evaluator, no `delta_recall_v1` synthetic harness — that exists only as a spec in `Evaluation-Plan.md:82-98`). What exists:
- Unit tests: `tests/memory/` — `test_websocket_memory.py` (session isolation, reset requirement, rollback, zero-mode no-commit, legacy reset), plus `test_recurrent_memory.py`, `test_memory_fusion.py`, `test_state.py`, `test_checkpoint_migration.py`, `test_qwen_token_selection.py`, `test_sequence_sampling.py`, `test_video_decode.py`.
- `deployment/model_server/debug_server_policy.py`: smoke-test client sending a synthetic observation over the wire (`:1-16` docstring; `deployment/model_server/README.md`).
- Training-time metrics only: JEPA predicted/target latent scalar stats + figures via `starVLA/training/trainer_utils/jepa_analysis.py`, logged from `train_vlajepa_cotrain.py:490-513`; memory diagnostics `memory/*` (`:462-468`). The in-training eval probe was deliberately disabled to keep dataloader positions deterministic — `_safe_eval` is now a barrier/log no-op stating "Full LIBERO evaluation is already run by the dependent evaluation job" (`train_vlajepa_cotrain.py:515-530`).

## MAP REPORT: docs-history

# VLA-JEPA: Project Intent and History

## 1. Base project / paper thesis

- The repo implements the paper "VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model" (arXiv:2602.10098, authors Jingwen Sun et al., 2026) — README.md:1-5, README.md:219-230. It is a fork/extension based on starVLA (README.md:77, README.md:214).
- Stack: Qwen3-VL-2B VLM backbone + V-JEPA2 ViT-L encoder (`facebook/vjepa2-vitl-fpc64-256`) as a latent world model, DiT flow-matching action head — README.md:84.
- Training data: SSV2 human video + LeRobot-format robot datasets (Droid, LIBERO, BridgeV2, Fractal) — README.md:89-97. Evaluation: LIBERO, LIBERO-Plus, SimplerEnv — README.md:138-196.
- Architecture fact central to the memory work: the control path (Qwen embodied tokens → DiT) and the world-model path (Qwen action-marker tokens → V-JEPA predictor) are **parallel, not serial**; `predict_action()` never calls the JEPA predictor, and `predicted_states` are not consumed by the action head even in training — docs/VLA-JEPA-Memory-Design-Proposal.md:31-37, figure docs/assets/memory/current_vlajepa_architecture.svg.

## 2. Memory research goal ("memv1")

The current research thrust is a **stateful VLA-JEPA extension**: a small, in-model, fully differentiable, bounded, explicitly state-passing memory bridge inserted after Qwen and before the action-head/world-model consumers — docs/VLA-JEPA-Memory-Design-Proposal.md:5, :11-27. Two-tier target:

| Tier | State | Status |
|---|---|---|
| Working memory | 8 recurrent slots `[B,8,512]`, gated cross-attention update | implemented (Stage 1 / memv1) — Design-Proposal.md:24-27, :170-182 |
| Episodic memory | gated-delta fast-weight matrix `[B,128,128]` FP32, learned half-life retention | deferred Stage 2, NOT implemented — Design-Proposal.md:25-27, :184-215; `long_term.enabled: false` in scripts/config/vlajepa_memv1_stage1.yaml:83-86 |

Design constraint from user memory file (persisted direction): memory must be "elegant/in-model/differentiable, not agentic (no FAISS/MemGPT)" — matches Design-Proposal.md:315 ("External vector database | reject for first version") and the original commit message of 8c1476d ("replacing the agentic FAISS/controller/MemGPT design").

## 3. Design history (git)

- `d14f5b9`/`4e29782` (first commits) → evaluation support commits (LIBERO 6e4846c…, LIBERO-Plus d5070d0…, SimplerEnv 84f1e9e/365b0c8) → README updates.
- `8c1476d` (2026-06-27) "Add elegant short/long-term memory design proposal": first 414-line proposal, "recurrent `[mem]` tokens + associative fast-weight matrix"; **proposal only, no code**. Its short-term tier lived **inside the JEPA predictor** with action-head conditioning optional (old doc §5.1-5.2; headers via `git show 8c1476d:docs/...`).
- `fad6297` (2026-06-30) "Add all-dataset cotraining and evaluation pipeline": cluster sbatch scripts, `vlajepa_cotrain_all.yaml`, data-prep for all robot datasets, export/eval scripts (~4600 insertions).
- `cbf7e98` (2026-06-30) "Replace memory design with implementation roadmap": rewrote Design-Proposal (618 lines changed), added Implementation-Plan (535 lines) and Evaluation-Plan (349 lines). The rewrite **explicitly repudiates the 8c1476d design** because predictor-only memory cannot affect deployed actions (`predict_action()` never runs the predictor) — Design-Proposal.md:29-47.
- `52ea8a2` then `a5e6516` (2026-06-30): Mermaid flows, then replaced by 5 academic SVG figures under docs/assets/memory/.
- `109439b` (2026-07-01) "Implement recurrent memory training pipeline": the actual memv1 implementation — new `starVLA/model/modules/memory/{state,recurrent_memory,fusion}.py` (103/279/100 lines), VLA_JEPA.py rewrite (+607/-…), segment sampling in datasets.py (+372), websocket server reset/session state (+133), 3 memv1 configs (`vlajepa_memv1_video/stage1/cotrain.yaml`), pipeline submit script `cluster/submit_vlajepa_memv1_pipeline.sh` (205 lines), and 8 test files (~875 test lines).
- `25e1882` (2026-07-01) "Handle corrupt robot videos deterministically": `VideoDecodingError` wrapping, bounded deterministic retry (`MAX_VIDEO_DECODE_ATTEMPTS`), tests/memory/test_video_decode.py.

**Documentation inconsistency:** README.md:208 still says "These documents describe a proposal and roadmap; the memory module is not implemented yet" and Design-Proposal.md:3 says "no memory implementation is present yet" — both stale as of commit 109439b, which implemented Phase 0/1.

## 4. Committed design constraints (architectural invariants)

Design-Proposal.md:73-85, all nine invariants: (1) train/deploy parity — memory must affect `predict_action()`; (2) causal writes only (never future frames, target actions, `gt_states`, predicted latents); (3) read-before-write (decision t reads M_{t-1}); (4) explicit state, never a model buffer; (5) per-sample reset masks; (6) no robot↔SSV2 cross-stream carry; (7) bounded cost independent of episode length; (8) `memory.enabled: false` is a byte-identical baseline; (9) side-effect-free forward under activation checkpointing.

Other hard commitments:
- Write source: Qwen **action-marker tokens** only (`source: qwen_action_tokens_current_only`) because they exist in all prompts and depend only on current images/task — Design-Proposal.md:93, memv1_stage1.yaml:59. Forbidden sources enumerated at Design-Proposal.md:226-232 (V-JEPA `input_states` are forbidden — the 8-frame window can contain post-decision information, Design-Proposal.md:44).
- Fusion: zero-gated (or documented small nonzero gate) residual cross-attention in a 2048→512→2048 bottleneck; exact no-op at init; concatenation rejected as primary because zero tokens still change softmax denominators — Design-Proposal.md:97-113. Actual stage1 config uses `gate_init: 1.0e-03`, `zero_init_gate: false` — memv1_stage1.yaml:67-68 (the documented small-nonzero option, Design-Proposal.md:111).
- Memory recurrence runs in an FP32/autocast-disabled island, kept FP32 even after `model.to(bfloat16)` for serving — Design-Proposal.md:213, Implementation-Plan.md:142.
- Training requires contiguous same-episode segments (`(dataset_id, episode_id, base_step_0..K-1)`), K=4 supervised decisions, same-episode burn-in for mid-episode starts, stride derived from replanning cadence (7 for LIBERO), no graph across `optimizer.step()` — Design-Proposal.md:243-266.
- Serving: per-WebSocket-connection `MemoryState` + private `torch.Generator`, real `type: reset` protocol, atomic commit/rollback, B=1 only — Design-Proposal.md:274-293.
- Checkpoints: memory params under `memory_module.*`/`policy_memory_fusion.*` allowlisted prefixes; everything else strict; warm start (not resume) from the allv2 100K checkpoint — Design-Proposal.md:297-305, Implementation-Plan.md:369-389; enforced in memv1_stage1.yaml:145-151.
- Parameter cap: first complete memory package < 10M learned params (<0.4% of the 2.77B model) — Implementation-Plan.md:520.
- Explicit non-goals for first release: no vector DB, no global episode buffer, no test-time inner-loop optimizer (Titans deferred), no imagined-state writes, no predictor RoPE/mask rewrite, no cross-batch recurrent graph, no claims from training loss alone — Implementation-Plan.md:524-532; Design-Proposal.md:215 (Titans out of scope until fixed-state capacity is proven the bottleneck).

## 5. Planned-but-unimplemented stages

Phased delivery (Implementation-Plan.md:391-472):
- **Phase 0** (contracts/no-op parity) and **Phase 1** (action-path working memory, robot-only warm start, frozen `qwen_vl_interface,vj_encoder,vj_predictor`… though memv1_stage1.yaml:154 gives qwen a 1e-6 LR entry while freeze_modules at :161 freezes it) — implemented in 109439b.
- **Phase 2 — long-term associative (gated-delta) memory**: NOT implemented. Requires exact hand-calculated delta-update test, ≥95% recall on versioned synthetic `delta_recall_v1` protocol, bounded norm over 10K writes — Implementation-Plan.md:434-447, Evaluation-Plan.md:84-98. `long_term.enabled: false` everywhere; no gated-delta code in `starVLA/model/modules/memory/__init__.py` (only `RecurrentMemory`, `ResidualMemoryFusion`, `MemoryState`, `MemoryRead`).
- **Phase 3 — world-model conditioning**: NOT implemented/enabled. Residual fusion into the 24 JEPA action tokens preserving the predictor's token contract, plus optional ordered SSV2 windows — Implementation-Plan.md:449-462, Design-Proposal.md:215-222 (§5.4-5.5 pseudocode at Implementation-Plan.md:215-225). `world_model_conditioning.enabled: false` in memv1_stage1.yaml:69-70 and memv1_cotrain.yaml:69-70; only `policy_memory_fusion.` is in the checkpoint-migration allowlist (memv1_stage1.yaml:148-150), `world_memory_fusion.*` is "optionally" allowlisted in Implementation-Plan.md:384.
- **Phase 4 — scaled training** ladder: 5K robot-only pilot → 10-20K cotrain → longer unroll sweep → full retrain only if warm-start evidence justifies it — Implementation-Plan.md:463-472.
- Postponed/rejected alternatives (Design-Proposal.md:306-318): Qwen prefix/KV memory (postponed, §5.2), predictor prefix tokens (postponed, §5.3 — breaks `ACRoPEAttention`'s `view(B, T, action_tokens + H*W, C)` contract, Design-Proposal.md:121), DiT-internal state (rejected), batched/multiplexed serving with `session_ids[B]` (postponed, Design-Proposal.md:282).
- The three memv1 runs form a dependency chain submitted by cluster/submit_vlajepa_memv1_pipeline.sh:1-27: `vlajepa_memv1_video` (memory **disabled**, `enabled: false` memv1_video.yaml:57; video pretrain seeded from `vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt`) → `vlajepa_memv1_stage1` (memory enabled, robot-only, warm-started from the memv1_video final model per memv1_stage1.yaml:144) → `vlajepa_memv1_cotrain` → eval job. The script requires a clean worktree at origin/main (submit_vlajepa_memv1_pipeline.sh:53-60).

## 6. Baseline numbers and acceptance gates

- Regression baseline: `VLA-JEPA-allv2-step_100000`, 10 trials/task, LIBERO-10 78%, Goal 92%, Object 98%, Spatial 96%, overall 364/400 = 91% — Evaluation-Plan.md:9-19. Cost anchors: 100K steps ≈ 51 h on 8×H100; 400-episode eval ≈ 21 min; model ≈ 2.77B params — Implementation-Plan.md:513-518.
- Framing: the question is not "beat 91%" (Object/Spatial near ceiling, most LIBERO tasks visually Markovian); it is (1) preserve baseline, (2) help when the cue leaves the observation, (3) zero/shuffle removes the gain, (4) reset isolates episodes — Evaluation-Plan.md:21-27.
- Nine runtime evaluation modes required per checkpoint (`build_disabled, runtime_bypass, live, zero, reset_each_decision, shuffle_within_batch, foreign_episode, short_only, long_only`) — Evaluation-Plan.md:31-44.
- Ablation matrix A0-A7 including a K-image-FIFO fixed-window baseline (A7) — Evaluation-Plan.md:154-165.
- Quality gates: +5-point paired LIBERO-10 dev gain to promote; final release needs 95% paired-difference CI excluding zero; non-inferiority lower bound > −2 points on standard suites; causal-use lower bound (live − bypass/foreign) > 0 with ≥3-point point difference; ≥10 points on a future versioned memory-dependent benchmark — Evaluation-Plan.md:254-263. Systems gates: <1 MB state/session, ≤10% p95 latency and step-time regression, ≤15% peak-memory regression — Evaluation-Plan.md:265-271. Stop conditions (falsification criteria) at Evaluation-Plan.md:336-349.
- Success criteria in Design-Proposal.md:320-332 (parity, reset independence, client isolation, future-perturbation invariance, live beats zero/shuffled, LIBERO non-inferior, <1MB & ≤10% latency).

## 7. Open questions the docs themselves raise

- No memory-dependent benchmark exists yet: "Standard LIBERO alone is insufficient"; candidates (delayed cue, occluded state, order memory, task progress, temporal interval, distractor resistance) must be built or staged; until a versioned benchmark manifest exists, "Level-C results are exploratory and cannot satisfy the +10-point gate" — Evaluation-Plan.md:127-138. LIBERO-Plus scripts are noted as not cluster-portable — Evaluation-Plan.md:136.
- Whether associative memory beats "a second set of slowly updated recurrent slots"; if not, keep two-timescale slots — Design-Proposal.md:215.
- Whether the claimed retention horizon (half-life 128-512 decisions) can even be trained given short unrolls — "training cannot claim that horizon unless its unroll/burn-in or auxiliary retrieval objective supplies credit at comparable delays" — Design-Proposal.md:211, Implementation-Plan.md:440.
- Stride/cadence generalization: 7 is LIBERO-specific; "must not assume every embodiment has the same FPS"; LIBERO replans once per 7-action chunk so memory advances per chunk — Design-Proposal.md:264, :293, Implementation-Plan.md:260.
- Whether to unfreeze the last Qwen blocks (~1e-6) if the memory/action-only run plateaus — Implementation-Plan.md:305.
- The historical 78/92/98/96 run is "a one-seed reference, not a confidence interval"; the paired comparator must be rerun — Evaluation-Plan.md:19, :146.
- Direct-context dropout (0.0 → 0.1 sweep) "only if needed" against memory collapse — Design-Proposal.md:260.
- Whether policy/world fusion projections should be shared — "share... only if an ablation supports it" — Implementation-Plan.md:520.
- Power analysis deferred: "Fifty trials per task is a planning value, not a guarantee of power" — Evaluation-Plan.md:243.

## 8. What the tests currently verify (tests/memory/, all unittest-style)

- **test_state.py:9-63** — `MemoryState`/`MemoryRead` contracts: `.detach()` preserves FP32/int64/bool dtypes; `.to(device)` preserves dtypes; TypeError on bf16 working tensor, ValueError on batch-size mismatch, TypeError on int32 steps; MemoryRead rejects bf16 tokens.
- **test_recurrent_memory.py:19-120** — init/read/write shapes and FP32 output under bf16 autocast; invalid rows read zeros, don't advance `steps`; write is out-of-place (different `data_ptr`); selective non-aliasing reset that can also activate a row; valid-mask zeroing incl. episodic; read-before-write (read returns previous slots); runtime state absent from `state_dict()` (:85-87); **delayed-gradient test** — loss at decision 1 produces finite nonzero grad on `source_projection` used by the write at decision 0 (:89-109); explicit shape/mask errors.
- **test_memory_fusion.py:12-87** — zero gate is an **exact** functional no-op (torch.equal) in fp32 and bf16; at zero gate the first gradient reaches only the scalar gate, adapter grad is exactly 0 (:30-46); small gate (1e-3) trains adapter; `bypass=True` returns the identical object; shape validation; default `RecurrentMemory`+`ResidualMemoryFusion` package < 10,000,000 params with normally-initialized (non-zero-std) weights (:80-87).
- **test_qwen_token_selection.py:9-24** — `VLA_JEPA._select_token_rows` picks marker positions per batch row; malformed marker count fails with a descriptive ValueError before any silent reshape.
- **test_sequence_sampling.py:102-362** — default mode preserves legacy single-step sampling; contiguous-segment mode never touches the pause-filtered `all_steps` (:121-130) and never uses global `random`; determinism keyed on (epoch, index, seed), val mode epoch-invariant (:208-224); supervised K decisions have exact stride, stay in-bounds with modality deltas, `segment_start`/masks correct, too-short episodes never selected (:226-251); burn-in is bounded, left-padded with `None`, `loss_mask=False` but `update_mask=True` (:253-282); negative delta handling; loud failure when no dataset has a valid segment; config validation; config plumbing through `get_vla_dataset` incl. `delete_pause_frame: false` (:322-362).
- **test_video_decode.py:30-68** — FFmpeg/AV errors wrapped as `VideoDecodingError` with video path; missing frames rejected before shape drift; plus in test_sequence_sampling.py:144-206, non-video failures stay fail-fast, video-decode retries are deterministic/shape-stable and bounded by `MAX_VIDEO_DECODE_ATTEMPTS`.
- **test_checkpoint_migration.py:27-80** — only `memory_module.` missing prefix accepted; disallowed missing key and any unexpected key rejected; `build_param_lr_groups` produces disjoint groups excluding frozen modules.
- **test_websocket_memory.py:74-164** — memory-enabled policy: infer before reset is rejected; two sessions are isolated with identical seeded outputs; failed inference rolls back memory and RNG atomically; `memory_mode="zero"` never commits state; oversized/failed reset is transactional; legacy `{"reset": true}` message supported; stateless policies may change batch size.

Not yet covered by tests (per the plan's own matrix, Implementation-Plan.md:474-509): gated-delta equation/stability, future-leakage integration test, two-rank DDP/Accelerate sharding-resume, old-100K end-to-end upgrade/export/reload, LIBERO smoke in live/zero/reset modes.

## Key file paths
- /lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/README.md
- .../docs/VLA-JEPA-Memory-Design-Proposal.md, VLA-JEPA-Memory-Implementation-Plan.md, VLA-JEPA-Memory-Evaluation-Plan.md
- .../docs/assets/memory/{current_vlajepa_architecture,memory_augmented_architecture,causal_memory_cycle,segment_data_pipeline,server_state_lifecycle}.svg
- .../scripts/config/vlajepa_memv1_{video,stage1,cotrain}.yaml; .../cluster/submit_vlajepa_memv1_pipeline.sh
- .../starVLA/model/modules/memory/{__init__,state,recurrent_memory,fusion}.py
- .../tests/memory/ (8 test modules listed above)

## MAP REPORT: evidence

# Empirical Evidence Inventory — VLA-JEPA memv1 memory module

Path abbreviations used below:
- `REPO` = `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA`
- `RUNS` = `/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs`
- `SLURM` = `REPO/outputs/slurm`

## 1. Prior LIBERO eval results (`REPO/results/`) — all NON-memory baselines; zero memv1 evals exist

Every eval: `eval_libero.py`, `num_trials_per_task=10` (100 episodes/suite), `seed=7`, `num_steps_wait=10`, `post_process_action=true`, `unnorm_key=franka`, single seed. Settings JSON printed at head of each `eval.log` (e.g. `REPO/results/libero_spatial/VLA-JEPA-cotrain-step70000/eval.log:11-32`).

| Checkpoint (`pretrained_path`) | with_state | spatial | object | goal | libero_10 |
|---|---|---|---|---|---|
| `RUNS/vlajepa_cotrain/checkpoints/VLA-JEPA-cotrain-step50000.pt` | true | **0.87** (`results/libero_spatial/VLA-JEPA-cotrain-step50000/eval.log:1088`) | **0.89** (`.../libero_object/.../eval.log:1008`) | **0.85** (`.../libero_goal/.../eval.log:978`) | **0.46** (`.../libero_10/.../eval.log:1075`) |
| `...VLA-JEPA-cotrain-step70000.pt` | true | **0.96** (`libero_spatial/...step70000/eval.log:1088`) | **0.96** (`libero_object/...:1008`) | **0.90** (`libero_goal/...:990`) | **0.65** (`libero_10/...:1099`) |
| `...VLA-JEPA-cotrain-step_82000.pt` | true | **0.83** (`:1088`) | **0.92** (`:1008`) | **0.91** (`:984`) | **0.46** (`:1090`) |
| `...VLA-JEPA-cotrain-step_100000.pt` | true | **0.80** (`:1089`) | **0.91** (`:1009`) | **0.88** (`:979`) | **0.41** (`:1073`) |
| `RUNS/vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt` | **false** | **0.97** (`:1089`) | **0.99** (`:1009`) | **0.96** (`:979`) | **0.72** (`:1064`) |

Per-task rates (10 tasks × 10 episodes) are in each eval.log as "Current task success rate" lines (e.g. allv2 libero_10 per-task: 0.8,0.8,1.0,0.8,0.2,1.0,0.7,0.8,0.3,0.8).

**Repeat-eval variance evidence** (same checkpoint, re-run):
- allv2-100K eval #1 `SLURM/vlajepa_evalall_6015834.out:17-20`: 10=0.78, goal=0.92, object=0.98, spatial=0.96
- allv2-100K eval #2 `SLURM/vlajepa_evalall_6016563.out:17-20`: 10=0.72, goal=0.96, object=0.99, spatial=0.97 (this run populated `results/`)
- cotrain-100K eval #1 `SLURM/vlajepa_eval100k_5922273.out:17-20`: 0.40/0.82/0.91/0.84 (10/goal/object/spatial)
- cotrain-100K eval #2 `SLURM/vlajepa_eval100k_6016562.out:17-20`: 0.41/0.88/0.91/0.80 (in `results/`)
→ suite-level noise up to ±0.06 at 100 episodes with identical checkpoint/seed setting.

Artifacts: `results/<suite>/<ckpt>/eval.log`, `server.log`, ~100 rollout MP4s named `rollout_<task>_<hash>_episodeN_{success,failure}.mp4`. Junk: `results/libero_object/VLA-JEPA-cotrain-step2000/` is an aborted 1-trial eval (server never connected; eval.log 132 lines of "Still waiting for server"); `results/libero_10 libero_goal libero_object libero_spatial/VLA-JEPA-cotrain-step50000/` is an empty shell-quoting-bug directory.

**No memv1 checkpoint has ever been evaluated on LIBERO** — `ls results/*/ | grep -i memv1` is empty; no `SLURM/vlajepa_evalrun_*.out` exists (that is the output pattern of `REPO/cluster/eval_after_run.sbatch:11`).

## 2. `RUNS/*/summary.jsonl` — schema and contents

Schema (all runs identical): one JSON object per checkpoint save, `{"steps": <int>, "time": <unix_seconds>}`. Nothing else (no losses).

- `RUNS/vlajepa_memv1_video/summary.jsonl`: 103 lines, `{"steps": 500, "time": 1782890833}` … `{"steps": 50000, "time": 1782928175}` (500-step interval, run completed 50K).
- `RUNS/vlajepa_memv1_stage1/summary.jsonl`: 11 lines, steps 500→5000 (500 interval; 5000 twice — final save + final_model). Gap 1000→1500 = 33461s (job crash/requeue window).
- `RUNS/vlajepa_memv1_cotrain/summary.jsonl`: 8 lines: 6988, 10000, 13911, 20000, 20847, 27752, 30000, 34729 — mix of 10K-interval saves and deadline saves; run in progress.
- `RUNS/vlajepa_memv1_smoke_109439b/summary.jsonl`: 1 line `{"steps": 1, "time": 1782890328}` (1-step pipeline smoke test, config `max_train_steps: 1`).
- Baselines: `vlajepa_cotrain` 213 lines (500→100000), `vlajepa_cotrain_allv2` 23 lines (8000→100000), `vlajepa_video` 103 lines (500→50000).

## 3. TensorBoard tags

### `RUNS/vlajepa_memv1_cotrain/tensorboard` (6 event files, hosts gpu-h100-{0183,0423,0303,0076,0307,0165}, one per 4h slurm segment)
39 scalar tags, **none memory-related**: `data_processed_decisions`, `loss_vla_action_loss`, `loss_vla_wm_loss`, `loss_vla_total`, `loss_vlm_wm_loss`, `loss_vlm_total`, `opt_epoch`, `opt_grad_norm_vla`, `opt_learning_rate`, `time_data`, `time_model`, `time_seconds_to_deadline`, plus 27 `jepa_*` diagnostics (`jepa_pred_gt_cosine_mean/p10/std`, `jepa_pred_gt_l1/l2`, `jepa_pred/gt_effective_rank`, `jepa_pred/gt_participation_ratio`, `jepa_pred/gt_feature_std`, `jepa_pred/gt_token_norm/variance`, `jepa_identity_baseline_cosine`, `jepa_pred_gain_over_identity`, `jepa_frac_tokens_cos_gt_0.9`, `jepa_effective_rank_ratio`, `jepa_feature_std_ratio`, `jepa_action_token_norm`, `jepa_action_to_state_norm_ratio`, `jepa_input_token_norm`, `jepa_view_view{0,1}_cosine`, `jepa_view_view{0,1}_pred_std`).

Representative values: `loss_vla_action_loss` 0.341@step10 → ~0.05–0.12 band by 7K → 0.087@38770; `loss_vla_wm_loss` 0.150@10 → 0.127@38770; `loss_vlm_wm_loss` ~1.47→1.34; `jepa_pred_gt_cosine_mean` 0.755@50 → 0.796@34700.

**Why memory tags are missing (bug, verified in code):** memory diagnostics are read in `REPO/starVLA/training/train_vlajepa_cotrain.py:462-468` *after* the VLM/video pass (`train_vlajepa_cotrain.py:441-452`); that pass calls `forward` on non-robot batches, which resets `self.last_memory_diagnostics = None` (`REPO/starVLA/model/framework/VLA_JEPA.py:419-425`, empty diagnostics when `has_actions` false, cf. `VLA_JEPA.py:382`). Robot-pass diagnostics set in `forward_sequence` (`VLA_JEPA.py:521-523`) get clobbered every step. Confirmed empirically: `grep -c "memory/" SLURM/vlajepa_cotrain_6029057.out` = **0**.

### `RUNS/vlajepa_memv1_stage1/tensorboard` (3 event files)
13 tags: the loss/opt/time tags plus **`memory_active`, `memory_policy_gate`, `memory_steps`, `memory_working_norm`** (robot-only stage, `skip_video_pass: true`, so no overwrite). Trends:
- `memory_policy_gate` (tanh of fusion gate, init 0.001): 0.00101@10 → 0.0135@1040 → plateau ≈0.01251@5000 (values at 1010/2000/3000/4000/5000: 0.0135/0.01284/0.01226/0.01251/0.01251).
- `memory_working_norm`: noisy 1.4–17.5 (9.95@1010, 17.51@3500, 1.50@5000).
- `memory_steps` (unrolled decisions incl. burn-in): 3–11; `memory_active`: constant 1.0.
- `loss_vla_action_loss`: 0.4399@10 → 0.0794@5000 (noisy: 0.195@3000, 0.253@3500).

### `RUNS/vlajepa_memv1_video/tensorboard` (3 event files)
35 tags (jepa_* + `loss_wm_loss`, `loss_total`, `opt_grad_norm`, no memory tags — `memory.enabled: false`). `loss_wm_loss` essentially flat: 1.4287@10 → 1.4716@19340 → 1.4025@38000 → **1.3861@50000**; `jepa_pred_gt_cosine_mean` 0.726@50 → 0.742@50000.

## 4. wandb
All three memv1 runs use online wandb: `project: vla-jepa`, `entity: crlc112358` (each `RUNS/<run>/config.yaml`). Run dirs: `RUNS/vlajepa_memv1_cotrain/wandb/wandb/run-2026070{1,2}_*-vlajepa_memv1_cotrain-E20260630-memv1-cotrain` (6 resumed segments, single run id `vlajepa_memv1_cotrain-E20260630-memv1-cotrain`); analogous for video (3 segments) and stage1 (3 segments). URL seen in logs: `https://wandb.ai/crlc112358/vla-jepa/runs/vlajepa_memv1_stage1-E20260630-memv1-stage1` (`SLURM/vlajepa_cotrain_6028729.err`). Known wandb data-loss on resume: "Tried to log to step 1010 that is less than the current step 1031 … data will be ignored" (`SLURM/vlajepa_cotrain_6028729.err`) — post-resume steps < high-water mark are dropped from wandb (tensorboard/JSON unaffected). `vlajepa_memv1_smoke_109439b/wandb/` is empty.

## 5. memv1 pipeline slurm logs (`SLURM/`)

**Manifest** `SLURM/vlajepa_memv1-109439b53008-20260701T072000Z.manifest`: git_sha=109439b53008 (commit "Implement recurrent memory training pipeline"), baseline_checkpoint=allv2-100K, video_job=6018626 → stage1_job=6018627 (afterok) → cotrain_job=6018628 → eval_job=6018629 (afterok, `eval_out_prefix=VLA-JEPA-memv1-live`, `eval_with_state=false`, `eval_memory_mode=live`, NUM_TRIALS=10; submit script `REPO/cluster/submit_vlajepa_memv1_pipeline.sh:191-194`).

**Smoke** (`SLURM/vlajepa_cotrain_6018602.out:2`): 1-step run of `train_vlajepa_cotrain.py` with stage1 config → `RUNS/vlajepa_memv1_smoke_109439b` (completed, has final_model).

**Video 50K** (`SLURM/vlajepa_video_6018626.out`, 3 requeue segments): loads allv2-100K (`line ~33`), freezes `vj_encoder,action_model`, 2770.333M total / **2289.180M trainable** (`vlajepa_video_6018626.out:49`), total batch 64, 50000 steps, SSV2 video only. Final: `loss/wm_loss` 1.3861@50000, "Training complete. Final model saved at RUNS/vlajepa_memv1_video/final_model" (tail of file). Key: wm_loss moved only 1.43→1.39 over 50K steps.

**Stage1 5K** (job name prefix `vlajepa_cotrain_`, 3 segments):
- `SLURM/vlajepa_cotrain_6018627.out:2` — `train_vlajepa_cotrain.py ./scripts/config/vlajepa_memv1_stage1.yaml`; loads `RUNS/vlajepa_memv1_video/final_model/pytorch_model.pt`; prints "loaded <full_model> with allowlisted new keys: ['memory_module.initial_slots', … 'policy_memory_fusion.output_projection.bias']" (31 new keys, `:102`); LR groups action_model/memory_module(16 params)/policy_memory_fusion(15 params) all 1e-4; freeze `qwen_vl_interface,vj_encoder,vj_predictor`; **161.501M trainable** (`:120`); total batch 8, "Supervised robot decisions per outer step = 32"; per-step log dict includes `memory/working_norm`, `memory/steps`, `memory/active`, `memory/policy_gate` (`:151-155`). Ran 0→~1040 then deadline.
- `SLURM/vlajepa_cotrain_6028729.err:65,98` — **crash at step ~1041**: `av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input: 'avcodec_send_packet()'` in DataLoader worker (rank5, worker 2) → whole job aborted. This motivated commit `25e1882` "Handle corrupt robot videos deterministically" (touches `starVLA/dataloader/gr00t_lerobot/{datasets.py,video.py}` + `tests/memory/test_video_decode.py`, `tests/memory/test_sequence_sampling.py`).
- `SLURM/vlajepa_cotrain_6029056.out` — resumed 1010→5000, completed; final step 5000: action_loss 0.0794, policy_gate 0.01251, working_norm 1.498; "in-training probe disabled to preserve deterministic dataloader position" printed at step 5000 (eval_interval is a no-op sync; `train_vlajepa_cotrain.py:519-530`); final model saved to `RUNS/vlajepa_memv1_stage1/final_model`.

**Cotrain** (`SLURM/vlajepa_cotrain_6029057.out`, still running): `vlajepa_memv1_cotrain.yaml`, loads stage1 final_model, freeze `qwen_vl_interface,vj_encoder`, **323.149M trainable** / 2776.652M total, max_train_steps=100000, `optimizer_steps_per_training_step: 2` (VLA pass + VLM/SSV2 pass), vlm loss_scale 0.1, `robot_world_model_loss: true`. Progress: step 38860 @ Jul 2 19:31 (~39% of 100K), ~1.8–2.2 s/step, ~7K steps per 4h requeue segment. Step logs contain `loss/vla_action_loss`, `loss/vla_wm_loss`, `loss/vlm_wm_loss` but **zero `memory/*` lines** (see §3 bug). Checkpoints: `RUNS/vlajepa_memv1_cotrain/checkpoints/{step_27752,step_30000,step_34729}`, plus a manual/premature export `VLA-JEPA-memv1-live-step_34729.pt` (7.47 GB, Jul 2 19:26) with symlink `VLA-JEPA-memv1-zero-step_34729.pt -> VLA-JEPA-memv1-live-step_34729.pt` (live/zero distinction is an eval-time flag, same weights). No `.training_complete` marker yet; `eval_after_run.sbatch:58-60` hard-requires it, so eval job 6018629 has not run.

**Key config facts** (from `RUNS/vlajepa_memv1_stage1/config.yaml` = `RUNS/vlajepa_memv1_cotrain/config.yaml` memory block): `memory.enabled: true`, `source: qwen_action_tokens_current_only`, `read_before_write: true`, `state_dtype: float32`; `action_conditioning: residual_cross_attention, bottleneck_dim 512, gate_init 0.001`; `short_term: 8 slots, dim 512, 8 heads, gated_cross_attention, update_gate_init 0.1`; `long_term.enabled: false`; `world_model_conditioning.enabled: false`; trainer `memory_bptt_steps: 4`, `memory_detach_burn_in: true`, `memory_direct_context_dropout: 0.0`; data `sample_mode: contiguous_segment, segment_length: 4, burn_in_max_decisions: 8, segment_stride: 7, require_same_trajectory: true`. memv1_video config has `memory.enabled: false`.

## 6. Evidence gap summary (exists vs missing for a memory-module paper)

**Exists:**
1. Baseline LIBERO success rates: 4 cotrain checkpoints (with_state=true) + allv2-100K (with_state=false, matching planned memv1 eval setting), 100 eps/suite, incl. 2 repeat evals quantifying eval noise (§1).
2. Stage1 memory-training curves: policy gate growth 0.001→0.0125 and action-loss 0.44→0.079 over 5K steps with memory active every step (tensorboard + slurm logs, §3/§5).
3. Full provenance chain: manifest with git SHA, config SHA-256s, parent checkpoints per stage (§5).
4. Loss/JEPA-representation diagnostics for all three memv1 stages (39/13/35 tags) and comparable baselines (`vlajepa_cotrain`, `vlajepa_cotrain_allv2` have their own tensorboard dirs).
5. Corrupt-video failure + deterministic-handling fix with tests (commit 25e1882).

**Missing:**
1. **Any task-level result for any memv1 checkpoint** (no LIBERO eval, no rollout videos, no success rates) — the single most important gap. Eval job blocked on cotrain completion (~39/100K steps, ETA several days at ~42K steps/day).
2. **Memory live-vs-zero ablation numbers** — infrastructure exists (`MEMORY_MODE` live/zero in `eval_after_run.sbatch:34,43`; zero symlink already created) but never executed.
3. **Memory diagnostics during cotrain** — `memory/*` tags absent from all cotrain logging due to the VLM-pass overwrite of `last_memory_diagnostics` (§3); gate evolution during the 100K cotrain is unrecoverable unless fixed and rerun/resumed.
4. No memory-stress benchmark (long-horizon/occlusion/recall tasks); only standard LIBERO suites are wired up.
5. No matched-step baseline-vs-memv1 comparison; single eval seed (7), 10 trials/task, with demonstrated ±0.06 suite noise — underpowered for small memory effects.
6. No in-training task probe in any memv1 stage (`eval_interval` intentionally a no-op, `train_vlajepa_cotrain.py:519-530`).
7. memv1_video stage shows near-flat wm_loss (1.43→1.39 over 50K steps) — no evidence this stage improves anything; no ablation skipping it.