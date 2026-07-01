# VLA-JEPA Memory Evaluation Plan

> Goal: determine whether memory is causally used, improves delayed-dependency behavior, preserves standard manipulation quality, and remains safe across resets and sessions.
>
> Architecture: [memory design](VLA-JEPA-Memory-Design-Proposal.md). Delivery sequence: [implementation plan](VLA-JEPA-Memory-Implementation-Plan.md).

## 1. Baseline to preserve

The completed all-data 100K checkpoint is the regression baseline:

| Suite | Successes | Episodes | Success rate |
|---|---:|---:|---:|
| LIBERO-10 | 78 | 100 | 78% |
| LIBERO-Goal | 92 | 100 | 92% |
| LIBERO-Object | 98 | 100 | 98% |
| LIBERO-Spatial | 96 | 100 | 96% |
| **Overall** | **364** | **400** | **91%** |

These results came from `VLA-JEPA-allv2-step_100000`, ten trials per task, with `WITH_STATE=false`. Treat them as a one-seed reference, not a confidence interval or a guaranteed reproducibility bound.

The first evaluation question is not “does memory exceed 91%?” Standard Object and Spatial suites are near ceiling and many LIBERO tasks are visually Markovian. The correct questions are:

1. Does memory preserve this baseline?
2. Does memory help when the required cue leaves the current observation?
3. Does zeroing or shuffling learned memory remove that gain?
4. Does reset prevent any previous episode or client from changing the next episode?

## 2. Evaluation modes required in the code

Every memory-enabled checkpoint must support the following runtime modes without changing learned weights:

| Mode | State behavior | Purpose |
|---|---|---|
| `build_disabled` | module absent under the old/disabled config | original architecture baseline |
| `runtime_bypass` | same trained checkpoint; skip memory read/write and fusion entirely | separates memory use from finetuned backbone/head changes |
| `live` | normal read/write/reset | proposed system |
| `zero` | hidden writes run, but the injected residual is forced exactly to zero | causal-use test without projection-bias/softmax artifacts |
| `reset_each_decision` | clear before every policy call | distinguishes recurrence from extra parameters |
| `shuffle_within_batch` | read another row's state in vectorized evaluation | tests content specificity when `B>1` is supported |
| `foreign_episode` | replay state from a matched but different episode | tests spurious history sensitivity |
| `short_only` | episodic tier disabled | tier ablation |
| `long_only` | working tier reset each call | tier ablation |

In the serial `B=1` LIBERO runner, use recorded matched foreign-episode state replay instead of `shuffle_within_batch`. Do not implement “reset disabled” as a normal benchmark setting. Use it only as a contamination diagnostic.

## 3. Correctness tests before policy evaluation

### 3.1 Disabled parity

- Load the existing 100K checkpoint with memory disabled.
- Use a per-session `torch.Generator` or identical pre-sampled initial flow noise; global RNG state is insufficient.
- Compare Qwen token extraction, action-head inputs, and normalized actions against the pre-memory commit.
- Require exact equality where deterministic or a documented numerical tolerance where kernels are nondeterministic.

### 3.2 Reset invariance

Run episode B in two ways:

1. Fresh server -> reset -> episode B.
2. Episode A -> reset -> episode B.

With identical episode-B inputs and per-session RNG, actions and memory diagnostics must match. The MVP server test uses two independent `B=1` connections; a separate module unit test covers selective reset for batched rows.

### 3.3 Client isolation

Open two WebSocket connections. Interleave distinct histories, reset one client, and prove neither client's outputs or state hashes depend on the other. Disconnect must release the corresponding state.

### 3.4 Future-leakage test

Construct two training segments identical through decision `t` but different afterward:

- different later camera frames.
- different future action labels.
- different JEPA targets.

With identical pre-sampled flow noise, the Qwen memory source, memory write, memory read, DiT conditioning tensor, and policy action at `t` must be unchanged. World tensors/loss may legitimately change when their explicit video context or target changes. This catches accidental policy-memory writes from full future video windows or labels without asserting that a world model ignores its valid inputs.

### 3.5 Delayed-gradient test

For a two-or-more-decision unroll, create a loss only on the final decision. Use the documented small-nonzero fusion gate (exact zero intentionally trains only the gate on its first backward), then confirm nonzero finite gradients on writer parameters used at an earlier decision. Repeat with a TBPTT boundary and confirm the expected detach.

### 3.6 Long-memory numerical tests

Version the synthetic protocol as `delta_recall_v1`:

- normalized 128-D Gaussian keys and 32 value classes with fixed unit-norm embeddings.
- 32 writes per sequence, 0-64 normalized distractor writes, and 25% same-key overwrites.
- training delays sampled uniformly from 8-64 decisions; held-out evaluation at delays 8, 32, 64, 128, and 256.
- 10,000 deterministic training sequences and 2,000 held-out sequences for each of five seeds.
- top-1 value-class accuracy from the memory read; chance is 3.125%.
- report in-range (<=64) and extrapolation (>64) results separately.

Before enabling gated-delta memory in a robot model:

- Match one update against a hand-calculated tensor result.
- Reach at least 95% held-out recall at trained delays on `delta_recall_v1`; do not apply this threshold to extrapolation delays.
- Run 10K writes without NaN/Inf and require the final-9K maximum state norm to remain below twice the first-1K p99 norm.
- Test same-key overwrite, distractor keys, reset, and half-life behavior.

## 4. Benchmark ladder

### Level A: fast development checks

- Unit and synthetic memory tasks.
- One LIBERO trial per task.
- One short training smoke run.
- Live versus zero memory on a fixed tiny set.

Purpose: catch contract, reset, shape, and stability bugs. These runs do not support model-quality claims.

### Level B: standard LIBERO development

Use the current four-suite evaluation with ten trials per task:

- 100 episodes per suite.
- 400 episodes total.
- fixed task initial states.
- fixed per-episode policy RNG.
- paired episode order across model variants.

Derive initial flow noise from `(eval_seed, suite, task, initial_state_id, decision_index)` or replay the same pre-sampled noise, so live/bypass/foreign comparisons are independent of request ordering and client interleaving.

Primary standard-suite metric: LIBERO-10 success. Goal is secondary. Object and Spatial are ceiling/regression checks.

### Level C: memory-dependent benchmark

Standard LIBERO alone is insufficient. Add or stage tasks in which an observation must be remembered after it disappears. At least one fixed benchmark should cover:

- **Delayed cue:** observe a color/object/instruction cue, endure a distractor interval, then act on it.
- **Occluded object state:** observe an object or drawer state before occlusion and use it later.
- **Order memory:** execute subgoals in a previously shown order.
- **Task-progress memory:** avoid repeating a completed subgoal when the final scene is locally ambiguous.
- **Temporal interval:** choose an action based on elapsed decision count or a prior event.
- **Distractor resistance:** preserve a relevant cue while unrelated objects enter the scene.

Candidates include a repaired LIBERO-Plus setup, a fixed partial-observability extension, or a dedicated delayed-cue harness. The current LIBERO-Plus scripts contain environment-specific paths and should be made cluster-portable before becoming a release gate.

Before any numerical release gate applies, publish a versioned benchmark manifest containing the exact task list, cue-delay buckets, initial-state IDs, episode limits, train/dev/final split, environment commit, and portable launch command. Until that artifact exists, Level-C results are exploratory and cannot satisfy the +10-point gate.

### Level D: final paired evaluation

For the winning architecture and a freshly rerun comparator:

- 50 trials per task on standard suites: 2,000 episodes total.
- Rerun the memory-disabled 100K baseline under the new per-session deterministic-noise protocol; the historical 78/92/98/96 run is not the paired comparator.
- Evaluate at least three independently trained memory seeds for the principal memory-dependent and standard-suite comparison.
- Paired task initial states and inference seeds.
- Per-task, per-suite, overall, and delay-bucket reporting.

## 5. Architecture ablations

Run the smallest matrix that answers one question at a time:

| ID | Working memory | Episodic memory | Action fusion | World fusion | Purpose |
|---|---|---|---|---|---|
| A0 | no | no | no | no | original 100K baseline |
| A1 | instantiated, zero gate | no | zero | no | checkpoint/parity control |
| A2 | yes | no | yes | no | action-path working-memory MVP |
| A3 | yes | no | yes | yes | world-model supervision value |
| A4 | no | yes | yes | no | episodic tier alone |
| A5 | yes | yes | yes | no | hierarchical policy memory |
| A6 | yes | yes | yes | yes | complete proposed model |
| A7 | no recurrent state; K-image FIFO | no | N/A | no | fixed-window history baseline |

For A7, place the last `K` decision images in temporal order in the Qwen prompt, reset the FIFO at every episode, and report actual tokens, parameters, FLOPs, and latency. If exact compute matching is impossible, call it a measured fixed-window baseline rather than “compute matched.”

Only after an architecture wins should capacity sweeps run:

- working slots: 4, 8, 16.
- working dimension: 256, 512, 1024.
- episodic key/value dimension: 64, 128, 256.
- segment length: 4, 8.
- retention half-life: 64, 128, 256, 512 decisions.

Avoid a full Cartesian product. Change one capacity axis at a time around the winning default.

## 6. Causal-use ablations

A higher live-memory score is not enough; extra parameters may be responsible. For every claimed gain, evaluate the same checkpoint with:

- live memory.
- zero memory.
- reset at every decision.
- matched foreign-episode state replay (`shuffle_within_batch` only for a future vectorized runner).

A credible memory result has all three properties:

1. Live memory beats zero/reset on memory-dependent tasks.
2. Shuffled or foreign memory does not match live memory.
3. The performance gap grows with cue-to-decision delay.

Report the action difference and success difference under ablation. Also report the norm of the residual memory injection; a nonzero state that is multiplied by a collapsed fusion gate is dead memory.

## 7. Metrics

### Policy quality

- Episode success rate.
- Success by task and suite.
- Success by task stage or subgoal.
- Success versus cue-to-decision delay.
- Repeated-subgoal and wrong-order failure counts.
- Action smoothness and action-chunk disagreement where relevant.

### World-model quality

- Existing predicted/target latent cosine and L1 metrics.
- Metrics bucketed by memory delay.
- Live-versus-zero memory delta on latent prediction.
- Multi-horizon latent accuracy if world fusion is enabled.

### Memory health

- Working-slot norm and pairwise cosine/effective rank.
- Update-gate distribution.
- Fusion-gate value and injected residual norm.
- Episodic state Frobenius norm and effective rank.
- Gated-delta write residual and retention half-life.
- Read similarity for correct, distractor, and foreign histories.
- Reset count and active-session count.

### Systems

- p50, p95, and p99 policy-call latency.
- GPU memory during inference and training.
- State bytes per session.
- Training examples/second measured in decision clips.
- Server session cleanup and maximum concurrent sessions.

## 8. Statistical reporting

For each suite and task:

- Report successes and total episodes, not percentages alone.
- Report 95% Wilson intervals as descriptive single-model summaries only.
- Use task-stratified paired analysis when variants use the same initial states. For multi-seed claims, use a hierarchical bootstrap that resamples training seed, task, then paired initial state, or a justified mixed-effects model.
- Never pool checkpoints and episodes as if every rollout were independent.
- Report per-seed results and the mean across training seeds.
- Do not choose the best seed after seeing final test results.

Predeclare LIBERO-10 as the sole primary standard-suite endpoint. Treat Goal, Object, Spatial, and per-task claims as secondary and apply Holm correction when making multiple significance claims. Use a fixed development split of task states for architecture decisions and reserve a separate final set when the benchmark permits it.

After the development pilot, estimate the observed discordant-pair rate and perform prospective paired-test power analysis before fixing final episode counts. Fifty trials per task is a planning value, not a guarantee of power for a five-point effect.

## 9. Acceptance gates

### Correctness gates

- Memory-disabled checkpoint and outputs preserve the baseline path.
- Future-leakage, reset, row-isolation, and client-isolation tests pass.
- No runtime state appears in a checkpoint.
- No robot state is reused by the SSV2 pass.

### Quality gates

- A +5-point paired LIBERO-10 development gain may promote a run to final evaluation.
- Final LIBERO-10 release requires a predeclared practically meaningful positive effect and a 95% paired-difference interval excluding zero.
- Standard-suite non-inferiority requires the lower 95% bound of the paired difference `(memory - rerun baseline)` to be greater than -2 percentage points.
- Causal-use requires the lower 95% bound of `(live - runtime_bypass/foreign)` to be positive, with a point difference of at least three points on the benchmark where a gain is claimed.
- Once a versioned memory-dependent benchmark exists, require a point improvement of at least ten points and a paired-difference interval excluding zero.
- Gains must persist across the predeclared training seeds; report an inconclusive result when intervals do not support the effect.

The numeric thresholds are release gates, not claims that every useful experiment must meet them. If memory helps only the delayed-cue benchmark and preserves standard LIBERO, that is still a valid result; report the scope honestly.

### Systems gates

- Runtime state remains below 1 MB per session.
- p95 policy-call latency regresses by no more than 10%.
- Matched-transition training step time regresses by no more than 10% in the first working-memory stage.
- Peak training memory regresses by no more than 15%, or the batch adjustment is documented.
- State is released on reset/disconnect and bounded under concurrent clients.

Measure latency on a fixed H100 model/dtype at `B=1`, after at least 20 warm-up calls and over at least 200 measured calls. Synchronize CUDA around model timing. Report model-only and end-to-end server p50/p95/p99, plus amortized time per control step because the current policy replans every seven environment steps.

## 10. Training-run ladder and approximate cost

Measured anchors:

- allv2 100K training: roughly 51 hours on eight H100 GPUs.
- 400-episode evaluation: about 21 minutes using the current multi-suite job.

Approximate matched-cost pilot ladder before memory overhead:

| Run | Steps | Approximate wall time | Approximate H100-hours |
|---|---:|---:|---:|
| smoke | 2K | ~1 hour | ~8 |
| pilot | 5K | ~2.6 hours | ~21 |
| medium | 10K | ~5.1 hours | ~41 |
| confirmation | 20K | ~10.2 hours | ~82 |
| full | 100K | ~51 hours | ~408 |

These are linear lower-bound planning numbers. Startup, data-cache, queue, and requeue overhead can dominate short runs, while freezing Qwen changes the slope. Because segment training processes multiple decisions per outer step, the estimates are meaningful only after normalizing by processed decision clips. Log that count explicitly.

Three 20K confirmation seeds are approximately 246 allocated H100-hours before overhead; three full 100K seeds are approximately 1,224 H100-hours.

Evaluation planning:

| Protocol | Episodes | Approximate wall time with current parallelism |
|---|---:|---:|
| four-suite development | 400 | ~21 minutes |
| four-suite final, 50 trials/task | 2,000 | ~1 hour 45 minutes |
| three memory checkpoints | 6,000 | ~5 hours 15 minutes sequential |
| three memory checkpoints + rerun baseline | 8,000 | ~7 hours sequential |

The 8,000-episode standard campaign is roughly 56 allocated H100-hours with the current eight-GPU request, before runtime-bypass, reset, and foreign-state ablations. Parallel jobs reduce wall time but not allocation cost.

## 11. Required result artifact

Every run should produce a machine-readable summary such as:

```json
{
  "checkpoint": "...",
  "checkpoint_sha256": "...",
  "config_sha256": "...",
  "memory_schema_version": 1,
  "memory_mode": "live",
  "training_seed": 42,
  "eval_seed": 7,
  "episode_seed": 7001003,
  "suite": "libero_10",
  "task": "...",
  "initial_state_id": "...",
  "episode_index": 0,
  "success": true,
  "episode_length": 187,
  "failure_reason": null,
  "latency_ms_p95": null,
  "state_bytes": null,
  "git_commit": "..."
}
```

This is an episode-level record; suite aggregates are separate derived records. Aggregate tables must be generated from episode records, not copied manually from terminal logs. Checkpoint/config hashes and exact initial-state identifiers are required, not optional placeholders in completed results.

## 12. Stop conditions

Stop scaling and revisit the design if any of these persist after a focused fix:

- Live, zero, and shuffled memory perform identically on delayed-cue tasks.
- Fusion gates remain near zero while direct observations solve the training loss.
- Working slots collapse to nearly identical vectors.
- Episodic state norm or rank becomes unstable.
- Reset contamination appears in any client or batch row.
- Gains disappear when future-window leakage is removed.
- Train/eval replanning cadence differs.
- Standard LIBERO regresses without a compensating memory-dependent gain.

The purpose of the evaluation is to falsify ineffective or unsafe memory designs early, not to justify a full run after the fact.
