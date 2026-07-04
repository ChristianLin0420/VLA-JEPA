# VLA-JEPA memv1 — Memory Experiment Campaign: Results & Analysis

**Period:** 2026-07-02 → 2026-07-04 · **Branch:** `memexp` · **Model:** Qwen3-VL-2B + V-JEPA2 + DiT action head + 8×512 recurrent working memory (6.3M params, residual cross-attention fusion, tanh-gated)
**Training:** video 50K → memory stage-1 5K → co-training 100K (complete). **Evidence:** wandb `crlc112358/vla-jepa` (groups `step_34729`, `step_100000`, `archaeology`, `droid_reach`) · raw records `results/**/episodes.jsonl`, `decisions.jsonl`, `results/offline/**/*.parquet`.

---

## 1. Executive summary

> **The memory module works — but not as a memory.** Across ten independent probes at two training stages, memv1's recurrent state functions as a **temporally-structured pacemaker**: the policy heavily exploits the state's maturity/dynamics statistics (injection ratio ≈ 0.46 of the fused input norm; removing the pathway costs −44 pp pooled success at 100K), while **episode-specific content is stored in the state (task decodable at 0.68) but never behaviorally read** (a different episode's memory substitutes perfectly, p=0.46 at n=400 paired episodes). The root cause is located in the task distribution — content is redundant with current observations on Markovian LIBERO — and *not* in the architecture (which stores) nor the optimization horizon (doubling BPTT/burn-in only sharpened the time signal). The one lever that remains is manufacturing content demand (occluded-cue tasks, T0.8).

All eval-time comparisons share weights, init states, and noise seeds (fully deterministic harness: five identical live repeats, episode-for-episode; auto-eval vs instrumented run episode-identical across checkouts).

---

## 2. Setup, provenance, infrastructure

- **Checkpoints:** step_34729 (38%, `ckpt_sha 2b981faac806`) and step_100000 (final). Archived anchors: stage1 {4500, 5000}; cotrain {30000, 34729, 40000+optimizer, 41686, 48668, 60000, 62555, 69467, 70000, 76460, 83336, 90309, 97254, 100000}. (Steps 5K–27.7K were pruned before archiving began — disclosed in Fig. 9 as a gap.)
- **Serve-time ablation surface** (hooks H1–H7, adversarially reviewed — 39 confirmed defects fixed pre-data): modes `live / prior / bypass / reset_k / freeze_k / write_every / foreign / noisematch / permute_once / noreset`, client blackout protocol, offline cache+replay engine (bit-exact λ=0≡bypass validation), consumed-episode enumerator (draw-for-draw parity-tested vs the production sampler).
- **Statistical protocol:** paired episodes (task, init state, noise seed) across modes; McNemar exact on discordants; offline Wilcoxon-scale n (≥5,600 paired segments). The harness replicate SD is exactly **zero** (determinism), so all inference rests on episode sampling; capture-context is held fixed across compared runs (ulp-level kernel caveat documented).
- **Terminology:** `prior` = the formerly-named "zero" mode (learned initial slots re-injected each decision — *not* a no-memory control); `bypass` = exact fusion skip (the true control).
- Two production bugs found and fixed en route: bf16 `.numpy()` crash that silently killed *all* evals; trainer clobber that erased all `memory/*` telemetry from co-training logs (telemetry restored from step 41.7K).

---

## 3. Results

### 3.1 Closed-loop causal grid — step_34729 (100 eps/suite/arm)

| arm | lib-10 | goal | object | spatial | pooled/400 | vs live (McNemar) |
|---|---|---|---|---|---|---|
| **live** | 2 | 43 | 63 | 40 | **148** | — |
| noreset (stale cross-episode state) | 2 | 43 | 65 | 48 | 158 | n.s. |
| noisematch (static moment-matched Gaussian) | 2 | 45 | 64 | 41 | 152 | n.s. |
| foreign (cross-task donor, 0 same-task) | 4 | 37 | 62 | 41 | 144 | n.s. |
| permute_once (slot roll at d=4) | 2 | 42 | 57 | 40 | 141 | n.s. |
| reset every 8 | 2 | 31 | 43 | 37 | 113 | 6e-5 |
| write every 4th | 0 | 33 | 10 | 22 | 65 | 1e-15 |
| freeze after 1 write | 0 | 34 | 5 | 17 | 56 | 8e-18 |
| **bypass** | 0 | 18 | 6 | 22 | **46** | 1e-18 |
| reset every 2 | 0 | 30 | 0 | 8 | 38 | 4e-25 |
| **prior** | 0 | 15 | 1 | 1 | **~17** | — |

**Reading:** every arm with *statistically mature* state ≈ live; every *immature*-state arm collapses toward/below bypass, ordered by immaturity. The pre-registered T0.4 falsification criterion (foreign ≈ live ≈ noisematch ⇒ statistics, not content) **fired**. Measured injection ratio ≈ **0.45** (‖tanh(g)·residual‖/‖consumer‖) — 20× larger than the raw gate (0.022) implies; per-decision counterfactual ‖Δaction‖ ≈ 0.58.

### 3.2 100K endpoint — the triad + content arms (100 eps/suite/arm)

| arm | lib-10 | goal | object | spatial | pooled/400 | vs live |
|---|---|---|---|---|---|---|
| **live** | 47 | 78 | 88 | 91 | **304** | — |
| **foreign** | 45 | 82 | 87 | 82 | **296** | p=0.46 — **content-null holds** |
| noisematch | 30 | 69 | 76 | 62 | 237 | p=5e-9 — **static noise no longer suffices** |
| bypass | 2 | 33 | 60 | 33 | 128 | p=2e-42 |
| prior | 0 | 0 | 1 | 6 | 7 | p=8e-90 |

**Reading:** (1) live performance matured strongly (37% → 76% pooled; libero_10 2% → 47%). (2) The **content-null survives full training**: another episode's memory still substitutes perfectly. (3) New at 100K: the head now rejects *static* surrogates — it reads the state's realistic **temporal evolution** (which foreign donors have and frozen Gaussians lack) — the "clock" refined into a trajectory-shaped pacing signal. (4) prior collapse is total (7/400): re-injecting initial slots is behaviorally catastrophic, the endpoint of the growing maturity-intolerance trend (§3.6). Injection ratio unchanged (0.458); counterfactual ‖Δaction‖ grew to 0.75.

### 3.3 Blackout stress test — step_34729 (goal+object /200; blackout = decisions 8..8+D, black frames, both views)

| arm | D=0 | D=1 | D=2 | D=4 |
|---|---|---|---|---|
| live | 106 | 82 | 67 | 51 |
| live, writes suppressed in gap | — | — | — | 55 |
| foreign donor | 99 | — | — | **62** |
| bypass | 24 | 23 | 13 | 18 |

The pathway keeps a **blind** policy above a **sighted** memory-less one (51–62 vs 24), but the bridging is content-free: foreign donors bridge equally well, and freezing writes during the gap changes nothing.

### 3.4 Offline teacher-forced battery — step_34729 (5,659 paired segments, 4 suites)

- Mode battery: bypass **+0.0081** MSE [CI +0.0075, +0.0088]; prior +0.036; **shuffled content −0.00002** (≈0; bound 400× below the pathway effect).
- Burn-in sweep: MSE 0.087→0.051, saturating at J=4; **forward ≡ reversed ≡ shuffled write order** to 4 decimals — pure write-count (integrator) signature.
- λ dose-response: minimum exactly at trained λ=1; live and shuffled-content curves identical at every λ; λ=0 reproduces bypass bit-exactly (validation contract).
- Depth profile: pathway benefit **front-loaded** (+0.019 at decisions 0–4 → +0.004 at 32–64) — a clock is most informative early.

### 3.5 Checkpoint archaeology (6 anchors, shared caches — directly comparable)

| ckpt | tanh(gate) | bypass ΔMSE | prior ΔMSE | content ΔMSE |
|---|---|---|---|---|
| stage1-5K | 0.0125 | +0.084 | +0.055 | −0.00007 |
| 30K | 0.0203 | +0.026 | +0.018 | +0.00003 |
| 40K | 0.0202 | +0.0056 | +0.024 | ~0 |
| 41.7K | 0.0205 | +0.0099 | +0.038 | +0.00001 |
| 48.7K | 0.0193 | +0.0060 | +0.035 | +0.00001 |
| 60K | 0.0197 | +0.0087 | +0.076 | +0.000003 |

**Born a clock** (content ≈ 0 always); offline pathway dependence squeezed 14× (stage1→40K plateau) while **maturity-intolerance grows monotonically** (prior +0.055 → +0.076 → behavioral 7/400 at 100K).

### 3.6 Probe battery — step_34729 (60 eps/suite; balanced accuracy, chance 0.10 / 0.20)

| target | memory | shuffled | const slots | present tokens |
|---|---|---|---|---|
| task identity | **0.68** | 0.09 | 0.09 | **1.00** |
| phase-of-episode | **0.63** | 0.19 | 0.20 | 0.77 |

**Content is stored but redundant**: the state linearly encodes the task at 7× chance, yet present tokens encode it perfectly — no gradient pressure to read memory content on Markovian suites. (Continuous-progress ridge rows returned unstable negative R² — a standardization artifact, excluded; classification rows carry the finding.)

### 3.7 DROID temporal reach (150 unseen episodes, enumerator-verified; 24.5K paired comparisons)

- **Sign flip off-distribution**: the injection *hurts* on DROID (bypass −0.0022 better at 0–11 decisions), decaying to ~0 by 100+. The pacing signal is a LIBERO-co-adapted shortcut with negative transfer.
- Content-null replicates on a second corpus to 150+ decisions.
- State perfectly bounded: working-norm saturates at 17.42 (p95 17.46) by decision ~12, flat to 150+ (12× the training horizon).

### 3.8 Credit-horizon branches (T2.3; 12K steps each from the same step_40000 warm start)

| arm (config) | bypass ΔMSE | prior ΔMSE | **content ΔMSE** | task-id decode | phase decode |
|---|---|---|---|---|---|
| reference (seg 4 / BPTT 4 / burn-in 8) | +0.0067 | +0.043 | −0.000002 [−5e-6,+2e-6] | 0.600 | 0.507 |
| **ctx16** (seg 8 / BPTT 8 / burn-in 16) | +0.0056 | +0.041 | +0.000021 [−2e-6,+4.9e-5] | 0.575 | **0.633** |

Doubling the credit window produced **zero content use** and **no additional stored content** — it built a *sharper clock* (phase decodability +12 pts). Optimization amplifies what the objective rewards; the objective on Markovian data rewards time only.

---

## 4. Unified interpretation

1. **What the module does:** integrates write-count/trajectory statistics into a bounded state whose maturity and evolution the co-adapted action head consumes as an episode-pacing signal (~46% of the fused input by magnitude; −44 pp pooled success when removed at 100K).
2. **What it does not do:** deliver episode-specific content to behavior — at 38% training, at 100%, offline at 2×10⁻⁵ resolution, under blackouts, on a second corpus, and under a doubled credit horizon.
3. **Why:** content is stored (0.68 decodability) but *redundant* — present observations already contain it (1.00). The only non-redundant signal in the state is time, so that is what gradient descent taught the head to read; longer training sharpened the requirement (static surrogates fail at 100K; maturity mismatch is catastrophic).
4. **Causal chain closed at every link:** architecture stores ✓ (probes) · optimization credits ✓ (T2.3 negative) · **data never asks ✗** — the redesign lever is task/content demand, not model or trainer.

## 5. Validity notes & pre-registered decisions

- **Determinism:** replicate SD = 0 (5 identical live runs; auto-eval episode-identical across checkouts). Seeds are the replication axis; all compared runs share the diagnostics-capture context.
- **Pre-registration honored:** the T0.4 falsification criterion fired and is reported as the headline; the co-primary live−prior / live−bypass endpoints are reported with paired McNemar as registered. Noise-calibration (k=5) re-scoped the power model to pure episode sampling.
- **Cross-harness checks passed:** λ=0 ≡ bypass (bit-exact single-forward); offline↔closed-loop orderings consistent at both checkpoints.

## 6. Limitations

- Single seed per training run (the T1.4 seed-noise floor was descoped when the content-null fired; loss-curve seed SD remains available from TB).
- `permute_once`: read-side permutation is architecturally a no-op (fusion reads are permutation-invariant) — the arm tests write-binding only.
- 10 trials/task per arm (n=400 pooled); adequate for the observed effect sizes (smallest reported significant gap p=5e-9), but subtle content effects ≤5 pp closed-loop cannot be excluded (offline bounds are 400× tighter).
- T2.3 branches ran 12K steps — emergence needing >12K steps under a longer window is not excluded, though the sharper-clock probe result argues against it.
- No memory-demanding benchmark exists yet in the suite (T0.8 unbuilt) — the content-read capability is untested under genuine demand; the present nulls diagnose the training distribution, not the ceiling.

## 7. Paper figure mapping

| Figure | Content | Source |
|---|---|---|
| Table 1 | Triads at 34.7K and 100K + content arms, paired CIs | §3.1, §3.2 |
| Fig. 2 | Memory vitals across training (gate, injection ratio, update-gate spread) | telemetry + archaeology |
| Fig. 3 | Offline dose-response: mode battery, burn-in order-invariance, λ curve | §3.4 |
| Fig. 4 | Maturity ladder (reset-K / freeze / write-cadence) | §3.1 |
| Fig. 5 | Transplant panel (foreign / noisematch / permute / noreset) at both checkpoints | §3.1, §3.2 |
| Fig. 6 | Blackout bridging + controls | §3.3 |
| Fig. 7 | Counterfactual divergence per decision; KM/hazard panels | decisions.jsonl |
| Fig. 8 | What memory stores (probe bars) vs what behavior uses | §3.6 |
| Fig. 9 | Emergence: bypass/prior/content ΔMSE + gate vs training step (pruned-window gap disclosed) | §3.5 |
| Fig. 10 | Temporal reach on DROID + state-norm stability | §3.7 |
| Table 2 | Credit-horizon branches (behavioral + representational) | §3.8 |

## 8. Recommended next steps

1. **T0.8 `libero_mem_v0`** (occluded-cue benchmark) — the decisive content-demand test; ships as an artifact regardless of outcome; the scene-edit helper and versioning scaffold exist.
2. **Train-under-demand** — once T0.8 exists, a branch trained with occlusion/dropout of present-token content (e.g., the dormant `memory_direct_context_dropout` knob, T2.4) is the constructive counterpart to the T2.3 negative.
3. **T1.1 no-memory twin** — now scoped as the *training-attribution* row only (does the pacing signal help training at all vs a twin?), 40K-step staged commitment.
4. Paper framing: "What does a differentiable memory learn when the data never asks? Anatomy of an episode clock" — mechanism + diagnosis + benchmark contribution.
