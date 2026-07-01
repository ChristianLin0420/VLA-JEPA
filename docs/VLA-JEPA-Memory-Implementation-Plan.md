# VLA-JEPA Memory Implementation Plan

> Scope: implementation roadmap for the architecture in [VLA-JEPA-Memory-Design-Proposal.md](VLA-JEPA-Memory-Design-Proposal.md).
>
> This document is intentionally staged. Do not implement predictor-prefix memory, associative long-term state, and a new sequence loader in one patch.

## 1. Definition of done

The first production-worthy memory release must provide:

- A memory-disabled path that is functionally identical to the current model.
- An explicit `MemoryState` input/output API.
- Contiguous, same-episode robot segments for differentiable unrolling.
- Working-memory reads that directly affect the DiT action path in training and inference.
- A real reset protocol and per-connection/session state isolation.
- Strict checkpoint migration from the allv2 100K model.
- Unit, DDP, checkpoint, serving, and LIBERO smoke tests.

Associative long-term memory and world-model conditioning are follow-up milestones, each gated by ablation results.

## 2. Proposed package layout

```text
starVLA/model/modules/memory/
  __init__.py
  state.py                 # MemoryState and MemoryRead dataclasses
  recurrent_memory.py      # working slots and optional episodic state
  fusion.py                # normally initialized, zero-gated bottleneck adapters

tests/memory/
  test_state.py
  test_recurrent_memory.py
  test_memory_fusion.py
  test_sequence_sampling.py
  test_checkpoint_migration.py
  test_websocket_memory.py
```

Keep state types separate from the learned module so serving and training code can manipulate reset masks without importing implementation internals.

## 3. Configuration schema

All missing memory keys must default to disabled.

```yaml
framework:
  memory:
    enabled: false
    schema_version: 1
    source: qwen_action_tokens_current_only
    read_before_write: true
    state_dtype: float32

    action_conditioning:
      enabled: true
      mode: residual_cross_attention
      bottleneck_dim: 512
      dropout: 0.0
      zero_init_gate: true

    world_model_conditioning:
      enabled: false
      mode: residual_cross_attention
      bottleneck_dim: 512
      dropout: 0.0
      zero_init_gate: true

    short_term:
      enabled: true
      num_slots: 8
      dim: 512
      num_heads: 8
      update: gated_cross_attention
      update_gate_init: 0.1

    long_term:
      enabled: false
      type: gated_delta
      key_dim: 128
      value_dim: 128
      retention_half_life: 256

datasets:
  vla_data:
    sample_mode: contiguous_segment
    segment_length: 4
    burn_in_max_decisions: 8
    segment_stride: ${framework.action_model.action_horizon}
    require_same_trajectory: true
    emit_episode_metadata: true
    delete_pause_frame: false

trainer:
  freeze_modules: "qwen_vl_interface,vj_encoder,vj_predictor"
  learning_rate:
    base: 1.0e-4
    memory_module: 1.0e-4
    action_model: 1.0e-4
  memory_bptt_steps: 4
  memory_direct_context_dropout: 0.0
```

Add `memory_schema_version` to the checkpoint sidecar and exported run config. Runtime memory contents are not checkpoint metadata.

## 4. Public state contract

Suggested types:

```python
@dataclass
class MemoryState:
    working: torch.Tensor
    episodic: torch.Tensor | None
    steps: torch.Tensor
    valid: torch.Tensor

    def detach(self) -> "MemoryState": ...
    def to(self, *, device) -> "MemoryState": ...


@dataclass
class MemoryRead:
    tokens: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
```

Required semantics:

- `working`: `[B, num_slots, memory_dim]`, stored/recurred in FP32.
- `episodic`: optional `[B, value_dim, key_dim]`, FP32; read is `episodic @ q` and the write is `(v - episodic @ k) k^T`.
- `steps`: per-row completed-policy-decision counter.
- `valid`: whether each row is an active, non-padding episode row.
- `MemoryRead.tokens`: previous working slots `[B,8,512]`, plus one projected long-read token when enabled, `[B,9,512]`.
- `.detach()` detaches every floating tensor together.
- `.to(device=...)` moves state without changing FP32/int64/bool dtypes.
- `init_state(batch_size, device)` expands learned working initialization, zeros episodic state and steps, and marks real rows valid.
- `state=None` initializes; reset applies before read to selected rows and returns non-aliased storage.
- Invalid padded rows produce zero loss and no read, write, or step increment; terminal rows become invalid after their final decision.
- Batch-size, row-identity, shape, or device mismatch without explicit reset is an error.
- State never appears in `named_parameters()`, `named_buffers()`, or `state_dict()`.

Memory recurrence and associative math execute with autocast disabled in FP32. Cast Qwen source tensors to FP32 on entry and only projected fusion residuals back to Qwen's dtype. Serving must keep or restore `memory_module` to FP32 after the existing blanket `vla.to(torch.bfloat16)` call.

Use one episodic orientation everywhere: `S [B,Dv,Dk]`. With normalized `q,k [B,Dk]`, `v [B,Dv]`, write strength `beta`, and retention `lambda`:

```text
read = S q
S_bar = lambda S
S_next = S_bar + beta (v - S_bar k) k^T
```

Pool the safe Qwen action-marker source to produce q/k/v/beta. Use `lambda = 2^(-delta_steps / half_life)` with a positive learned half-life. Project the `[B,Dv]` read to one `[B,1,512]` token before appending it to the working-slot read bank.

## 5. Framework refactor

### 5.1 Factor Qwen token extraction

In `starVLA/model/framework/VLA_JEPA.py`, extract the duplicated Qwen logic into a helper used by both `forward()` and `predict_action()`:

```python
def _encode_qwen_tokens(images, instructions, prompt_template):
    # returns last_hidden, action_tokens, embodied_action_tokens
```

Robot `forward()` and `predict_action()` must return:

- `action_tokens [B, N_action, 2048]`, where `N_action` is derived from latent context bins times markers per bin and resolves to 24 under the current config.
- `embodied_action_tokens [B, 32, 2048]` for robot/action prompts.

SSV2/video-only forward has action markers but no embodied markers, so its helper result must permit `embodied_action_tokens=None`. Assert configured token counts with descriptive errors. Silent `.view(B, -1, H)` behavior can hide missing markers.

### 5.2 Add a single-decision pure function

Factor the current forward into a single-decision function:

```python
def _forward_one(example_batch, memory_state, reset_mask, update_memory):
    state_before = memory.reset_state(memory_state, reset_mask)
    qwen = _encode_qwen_tokens(...)
    read = memory.read(qwen.action_tokens, state_before)  # [B,8 or 9,512]

    policy_tokens = policy_fusion(qwen.embodied_action_tokens, read.tokens)
    world_tokens = world_fusion(qwen.action_tokens, read.tokens)

    losses = _compute_action_and_world_losses(policy_tokens, world_tokens, ...)
    state_after = memory.write(qwen.action_tokens, state_before) if update_memory else state_before
    return losses, state_after, read.diagnostics
```

The write occurs after all prediction tensors are formed. Do not mutate `state_before` in place. Only sequence forward may return graph-bearing state; stepwise training must detach before an optimizer boundary and inference always returns detached state privately to the server.

### 5.3 Preserve the old return type

The current trainers sum every value returned by `model(batch)`. Returning state in that dictionary would be a correctness bug. Use one of these interfaces:

- Keep `forward()` returning only the loss dictionary and add `forward_sequence()` for stateful training; or
- Return a typed output and update every trainer before enabling memory.

The first option is lower risk and preserves old callers.

### 5.4 Action-path fusion

Add the residual cross-attention adapter before repeated diffusion-step expansion:

```python
q = query_down(layer_norm(embodied))              # [B,32,512]
delta = query_up(attend(q, memory_read.tokens))   # [B,32,2048]
conditioned = embodied + tanh(gamma_policy) * delta
embodied_repeated = conditioned.repeat(repeated_diffusion_steps, 1, 1)
```

Initialize adapter weights normally and `gamma_policy=0`; zeroing both would make the branch gradient-dead. Exact parity is tested at zero gamma, where only gamma receives the first gradient. The delayed-writer gradient test and Phase-1 training use a documented small nonzero gate or a short gate-only warm-up. The same fusion must run once before `FlowmatchingActionHead.predict_action()`. Never update memory inside the DiT denoising loop.

### 5.5 World-model fusion

Stage 3 can condition the predictor while keeping its 24-token contract:

```python
q = world_query_down(layer_norm(actions))
delta = world_query_up(attend(q, memory_read.tokens))
conditioned_actions = actions + tanh(gamma_world) * delta
predicted_states = self.vj_predictor(input_states, conditioned_actions)
```

Do not add global prefix tokens to `vj_predictor.py` in the initial implementation.

## 6. Contiguous robot-segment sampling

### 6.1 Current problem

`LeRobotMixtureDataset.sample_step()` randomly chooses a dataset and one step. The returned example omits dataset identity, episode identity, base index, and boundaries. Consecutive DataLoader rows are unrelated.

### 6.2 Required sample

Add a `sample_segment()` path that returns `K` ordered decisions from one dataset and trajectory:

```python
{
    "burn_in": [example_b0, ..., example_bJ_minus_1],
    "sequence": [example_t0, ..., example_tK_minus_1],
    "dataset_id": str,
    "episode_id": int,
    "segment_start": bool[K],
    "base_indices": int64[K],
    "is_first": bool[K],
    "is_last": bool[K],
    "sequence_valid": bool[K],
    "loss_mask": bool[K],
    "update_mask": bool[K],
}
```

The composite `(dataset_id, episode_id)` is required because episode indices repeat across datasets. `segment_start` is not the same as the true episode `is_first` flag. A mid-episode segment requires a same-episode burn-in prefix; burn-in updates state without supervised loss and may be detached before the supervised `K` decisions.

### 6.3 Sampling rules

- Preserve the intended mixture by choosing the dataset with the configured dataset probability, then a trajectory/valid start with an explicitly documented weighting policy. Do not accidentally change trajectory-length bias when switching from steps to starts.
- All decision windows must fit inside the same raw trajectory.
- Base indices are raw trajectory indices, monotonic with the configured dataset-specific stride; never advance by positions in filtered `all_steps`.
- Derive stride from action horizon/replanning cadence and dataset timing metadata. It resolves to seven for the current LIBERO evaluation but must not assume every embodiment has the same FPS.
- Set `delete_pause_frame: false` initially and explicitly plumb it through `build_dataloader -> get_vla_dataset -> make_LeRobotSingleDataset`. If filtering is later enabled, prove temporal continuity.
- Tail padding uses repeated/zero payload only where required by collation, with `sequence_valid=false`, `loss_mask=false`, and `update_mask=false`; padded rows never advance memory.
- Include the segment configuration in every cache fingerprint. Remove hard-coded cache filenames that bypass configuration hashes.
- Determinism must depend on `(epoch, unique global sample index, seed)`. Let Accelerate shard global samples; adding rank to the seed makes results world-size-dependent.

### 6.4 Collation

Collate burn-in plus supervised decisions into batch-major sequences. The model may flatten `B × (J+K)` for frozen/heavy encoders, then reshape outputs and recurrently process time. Do not maintain a rank-local dictionary of episode states across batches. Test actual Accelerate sharding and exact resume, not only raw DataLoader determinism.

## 7. Training-loop integration

### Robot pass

1. Initialize one state per segment batch.
2. Reset only true episode starts; reconstruct a mid-episode state with its same-episode burn-in prefix.
3. Vectorize Qwen/V-JEPA encoding over `B × (J+K)` where memory permits.
4. Run burn-in read/write with no supervised loss, optionally detach, then process the `K` supervised decisions.
5. Sum action loss over `loss_mask & sequence_valid`, divide by the valid-decision count, and preserve any explicit world-loss scale.
6. Perform one backward and optimizer update for the complete segment.
7. Detach only at a declared TBPTT boundary; no graph crosses `optimizer.step()`.
8. Log processed decision clips and drive run comparisons/schedules with that count.

### SSV2 pass

Stage 1 uses a robot-only trainer and skips the SSV2 optimizer pass and world-model loss. With Qwen and `vj_predictor` frozen, the existing SSV2 backward would have no trainable path and could fail. Passing robot state into any later SSV2 pass is forbidden.

If world-model memory is later trained on SSV2, add ordered multi-window samples from a single video and keep a separate state object for that pass.

### Optimizer groups

For the first warm-start run:

| Group | Suggested setting |
|---|---:|
| V-JEPA encoder | frozen |
| Qwen | frozen initially |
| JEPA predictor | frozen in action-only Stage 1 |
| Action head | current action-head LR or a reduced warm-start LR |
| Memory and fusion adapters | `1e-4` initial sweep |

Use exact freeze paths `qwen_vl_interface,vj_encoder,vj_predictor`. Put `memory_module` and `action_model` under `trainer.learning_rate`; the current optimizer builder does not consume a standalone `trainer.memory_lr`. Add an optimizer-group test that every trainable parameter appears exactly once. Unfreeze the last Qwen blocks at approximately `1e-6` only if the memory/action-only run plateaus and evaluation indicates a representation bottleneck.

### Logging

Add diagnostics through the existing JEPA/W&B side channel, but do not add them to the loss dictionary:

- working-slot norm and pairwise cosine.
- update-gate mean, p05, and p95.
- policy/world fusion gate values.
- memory read and residual-injection norms.
- episodic norm, effective rank, retention, and delta residual.
- fraction of reset rows and valid sequence decisions.

## 8. Serving and reset implementation

### 8.1 Protocol

Support explicit messages:

```json
{"type": "infer", "request_id": "...", "payload": {"batch_images": [], "instructions": []}}
{"type": "reset", "request_id": "...", "payload": {"episode_id": "...", "episode_seed": 7}}
```

The MVP allows one active `B=1` episode per connection and rejects live-memory requests with a different batch size. Multiplexed/batched inference requires stable `session_ids[B]`, row-level states, reset masks, and explicit row-reordering semantics and is postponed.

### 8.2 Ownership

Preferred ownership is per WebSocket handler:

```text
connection opens -> state = None
reset(seed)       -> state = None; generator = Generator(seed)
infer             -> public_output, candidate_state = policy.predict_action(...)
success           -> atomically commit candidate_state and RNG advancement
failure           -> keep state/RNG unchanged or invalidate the connection
disconnect        -> delete state and generator
```

`MemoryState` is never inserted into the public output dictionary or msgpacked. Pass a per-connection `torch.Generator` or pre-sampled initial flow noise through `VLA_JEPA.predict_action()` into `FlowmatchingActionHead.predict_action()`; global `torch.manual_seed` is not client-isolated.

### 8.3 Existing files to fix

- `deployment/model_server/tools/websocket_policy_server.py`: implement reset and connection-local state.
- `deployment/model_server/tools/websocket_policy_client.py`: send a typed reset request and validate the response.
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`: accept a generator or explicit initial action noise.
- `examples/LIBERO/model2libero_interface.py`: call `self.client.reset()` from `reset()`.
- SimplerEnv inference wrapper: apply equivalent episode lifecycle behavior.

### 8.4 Complete file-change map

| File or package | Required change |
|---|---|
| `starVLA/model/modules/memory/*` | new state, working/episodic update, and bottleneck fusion modules |
| `starVLA/model/framework/VLA_JEPA.py` | factor token extraction; add explicit sequence state, read/write order, policy/world fusion, private inference state return |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | accept per-session generator or explicit initial flow noise |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | raw-index segment and burn-in sampling, metadata, masks, cache fingerprint |
| `starVLA/dataloader/lerobot_datasets.py` | plumb sequence and pause-deletion configuration |
| `starVLA/dataloader/__init__.py` | stateful segment collate/loader configuration |
| `starVLA/training/train_vlajepa_cotrain.py` | robot-only sequence pass, masked normalization, diagnostics, processed-decision counter |
| `starVLA/training/trainer_utils/trainer_tools.py` | optimizer/freeze tests and allowlisted warm-start support |
| `starVLA/training/trainer_utils/run_utils.py` | memory schema/version in checkpoint sidecar |
| new migration command plus `base_framework.py` | memory-enabled config override and strict mismatch allowlist |
| `cluster/export_vlajepa_ckpt.py` | verify memory schema/keys on export; no runtime-state export |
| WebSocket server/client | reset, private connection state, RNG isolation, atomic commit |
| LIBERO/SimplerEnv interfaces | episode reset/seed plumbing and stable `B=1` contract |
| `examples/LIBERO/eval_libero.py` | episode seeds and machine-readable paired result records |

## 9. Checkpoint migration

### Disabled config

When `framework.memory.enabled` is absent or false:

- Do not instantiate any memory parameter.
- Preserve strict loading of the original allv2 100K checkpoint.
- Preserve the exact current forward and inference output shapes.

### Enabling memory from the 100K model

Add a one-time upgrade loader or command that:

1. Builds the memory-enabled model.
2. Loads the old checkpoint with a mismatch allowlist.
3. Permits only missing keys under `memory_module.*`, `policy_memory_fusion.*`, and optionally `world_memory_fusion.*`.
4. Rejects every unexpected key and every other missing key.
5. Creates a new memory-enabled run directory, copies `dataset_statistics.json`, writes the memory-enabled config, and places the upgraded `.pt` under that run's `checkpoints/` so `read_mode_config()` resolves the correct files.
6. Runs a zero-gate parity test before accepting the upgrade.

This is a warm start with a new optimizer, not an in-place resume of the old Accelerate full state. The existing export script can continue exporting the complete model state once learned parameters are present, but it should verify the declared memory schema and key prefixes. Runtime state is never exported.

## 10. Phased delivery

### Phase 0: contracts and no-op parity

Deliver:

- Disabled configuration schema.
- State dataclasses and pure reset helpers.
- Qwen token-extraction helper.
- Zero-gated fusion module.
- Reset RPC and session ownership.
- Checkpoint upgrade allowlist.

Gate:

- Old 100K strict-loads with memory disabled.
- Disabled and zero-gated outputs match baseline within deterministic tolerance.
- Exact parity either bypasses the adapter or uses `dropout=0`, so it does not consume RNG before flow-noise sampling.
- The zero-gated parity test gives a nonzero first gradient to the scalar gate; a separate small-nonzero-gate test gives delayed gradients to adapter and writer parameters.
- FP32 state remains FP32 under training autocast and BF16 server model casting.
- Runtime state is absent from `state_dict()`.
- Reset and two-client isolation tests pass.

### Phase 1: action-path working memory

Deliver:

- Contiguous robot segments.
- Same-episode burn-in for mid-episode segment starts.
- Eight gated recurrent slots.
- `K=4` supervised decisions with stride derived from deployment cadence (seven for current LIBERO).
- Direct action-path residual fusion.
- Robot-only warm-start pilot.

Gate:

- Same-trajectory/stride invariants pass across workers and ranks.
- Gradients reach a write from an earlier decision.
- Future perturbations cannot alter an earlier action.
- A 2K-step smoke run has finite losses, gates, and norms.
- Step time grows by at most 10% and peak memory by at most 15% for a matched transition count.

### Phase 2: long-term associative memory

Deliver:

- FP32 128×128 gated-delta state.
- Bounded retention parameterization.
- Long-read action fusion and diagnostics.
- A training source with credit at the claimed delay: longer unroll/burn-in, a versioned retrieval auxiliary, or synthetic pretraining.

Gate:

- Exact hand-calculated delta update test.
- More than 95% recall on the versioned synthetic protocol defined in the evaluation plan; report in-range and extrapolation delays separately.
- Bounded norm after 10K synthetic writes.
- Live episodic memory beats zero/shuffle on a long-delay task.

### Phase 3: world-model conditioning

Deliver:

- Residual memory fusion into the existing 24 JEPA action tokens.
- Optional ordered SSV2 windows.
- Independent robot and video state objects.

Gate:

- Predictor shape, mask, and RoPE paths remain unchanged.
- Long-lag latent metrics improve without policy regression.
- No target/future-derived tensor enters the memory writer.

### Phase 4: scaled training

Suggested sequence:

1. 5K robot-only action-memory pilot.
2. 10K-20K co-training run after passing evaluation gates.
3. Longer supervised-unroll/burn-in sweep if memory benefit grows with delay.
4. Full retraining only after warm-start evidence justifies the cost.

Compare runs by processed decision clips, not outer training steps, because segment length changes work per step.

## 11. Test matrix

### Unit tests

- `B=1` and `B>1` state/read/write shapes.
- Device and dtype conversions.
- State initialization, invalid padding, terminal invalidation, and shape-mismatch errors.
- Selected-row reset isolation.
- Read-before-write ordering.
- Zero-gate functional parity.
- Nonzero first gradient for the chosen zero-gate parameterization.
- Working-slot gate bounds and non-collapse initialization.
- Gated-delta equation and 10K-update stability.
- Detach/TBPTT behavior.
- Runtime state absent from model serialization.
- Token-count assertions.
- Future-frame and target-label leakage test.

### Data tests

- Same dataset and trajectory across a segment.
- Exact stride and monotonic base indices.
- No boundary crossing.
- Determinism across restarts.
- Distinct worker/rank samples without row-identity assumptions.
- Cache invalidation when sequence settings change.

### Integration tests

- One stateful VLA step followed by an unrelated stateless SSV2 step.
- Two-rank DDP forward/backward and save/resume.
- Actual Accelerate sharding and exact-resume segment sequence.
- Old 100K upgrade, save, export, and reload.
- WebSocket infer/reset/infer parity.
- Two simultaneous clients with independent histories.
- One-trial-per-task LIBERO smoke in live, zero, and reset modes.

## 12. Resource anchors

Measured repository baselines provide planning bounds:

- allv2 100K training used roughly 51 hours on eight H100 GPUs.
- The final logged model step took about 1.61 seconds plus 0.62 seconds of data time.
- The current model is about 2.77B parameters; `final_model/pytorch_model.pt` is about 6.83 GB, the exported tied evaluation `.pt` about 7.45 GB, and a full optimizer checkpoint is much larger.
- The 400-episode allv2 evaluation completed in about 21 minutes.

Eight FP32 512-dimensional slots plus one FP32 128×128 associative state use about 80 KiB per session. Use 2048→512→2048 bottleneck fusion, share policy/world projections only if an ablation supports it, and report the exact parameter count. Cap the first complete working+episodic+two-fusion implementation at 10M learned parameters (under 0.4% of the 2.77B model); do not silently use full 2048-wide attention. Naive segment unrolling can multiply activation cost by `K`; vectorize the heavy encoders over `B × (J+K)` before optimizing model capacity.

Tests require an explicit runner. Add `pytest` plus CPU/GPU/integration markers to development dependencies and configuration, or use the built-in `unittest` runner consistently; do not leave the proposed suite unexecutable.

## 13. Explicit non-goals for the first release

- No external vector database.
- No model-global episode buffer.
- No test-time inner-loop optimizer.
- No imagined state written into factual memory.
- No predictor RoPE/mask rewrite.
- No cross-batch recurrent graph.
- No claim of improvement based only on lower training loss.

The first release succeeds when memory is causal, isolated, deployable, and measurably useful—not when it contains the largest possible mechanism.
