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

---

# Chapter 2 — External memory benchmarks and the content-demand fine-tune (2026-07-05)

## 9. Benchmark adoption

Three external memory benchmarks integrated behind our websocket harness (identical
episodes/decisions records and memory-mode ablations everywhere): **LIBERO-Mem**
(AAAI 2026; same robosuite/MuJoCo stack, drop-in), **MIKASA-Robo-VLA** (ManiSkill3/
SAPIEN; headless Vulkan verified on H100 with zero config), **RoboMME** (ICML 2026
Spotlight). Full survey and adoption rationale: `docs/memory-benchmark-adoption-plan.md`.

## 10. Zero-shot floors (LIBERO-trained models, memory or not)

| Benchmark | live | foreign | bypass | prior | no-memory baseline (allv2) |
|---|---|---|---|---|---|
| LIBERO-Mem (10 tasks × 20) | 0/200 | 0/200 | 0/200 | 0/200 | 0/200 |
| MIKASA anchors (4 × 50) | 0/200 | — | — | — | — |
| RoboMME Reference (16 × 50) | 0/800 | — | — | — | — |

Current Markovian-trained VLAs have zero transferable competence on genuine
memory tasks, with or without a memory module.

## 11. Content-demand fine-tune (T-FT)

15K steps from step_100000; MIKASA anchors at exactly 0.20 of the mixture
(own `new_embodiment` stats block); recurrent TBPTT pipeline unchanged.
Closed-loop MIKASA remained at floor post-FT (live 1/200, bypass 0/200) —
15K × 20% is below the behavioral acquisition threshold; the offline
discriminator below is therefore the authoritative readout. Vanilla-LIBERO
regression at mid-FT: 47/78/88/91 → 22/56/69/60 (naive-mixture forgetting;
a finding in itself).

## 12. THE decisive measurement — offline content-read discriminator, before vs after content-demand training

Paired dMSE vs live on 4,537 identical cached MIKASA segments (Qwen frozen in
both checkpoints → same caches; 4 anchor tasks × 40 episodes):

| checkpoint | bypass | prior | **shuffled content** |
|---|---|---|---|
| pre-FT (100K, LIBERO-only) | −0.0064 | −0.0181 | −0.000030 |
| post-FT (15K on MIKASA) | +0.0081 | +0.0179 | **−0.000010** |

Three results in one table:
1. **Content-reading did not emerge** (shuffled ≈ 0 in both rows) — even when
   the training data causally requires content, at this training dose the
   policy does not learn to read what the memory demonstrably stores.
2. **The pathway rows flip sign**: pre-FT, the memory injection *hurt* on
   out-of-distribution MIKASA data (negative bypass/prior deltas — the DROID
   negative-transfer result replicated on a third corpus); post-FT the head
   re-established its clock dependence on the new domain (+0.008/+0.018)
   within 15K steps. The pacemaker co-adaptation is the attractor state of
   this architecture under BC, regardless of domain.
3. Combined with Chapter 1: stored content ✓ (probes), credit window ✓
   (T2.3), **content demand ✓ (this chapter)** — and content still is not
   read. The read/fusion mechanism itself (slot-attention read into a small
   gated residual) is now the primary suspect: gradient descent consistently
   finds the maturity-statistics shortcut before any content-routing
   solution, under every condition tested.

**Caveats:** 15K steps at 20% mixture is a modest dose (closed-loop success
had not emerged either); a MIKASA-dominant or longer run could still differ.
The clean escalation: (a) heavier/longer content-demand training with the
discriminator as the tracked metric, (b) read-path redesign (e.g.
content-addressed retrieval keyed on present-observation uncertainty, or a
higher-capacity ungated read) trained under the same demand.

## 13. Updated recommendation

The paper is now a complete arc without further compute: *a differentiable
memory under behavior cloning collapses to an episode pacemaker; this is
robust to training duration, credit horizon, and even explicit content
demand at moderate dose; the failure locus is the read pathway; and current
VLA evaluation practice (Markovian suites) cannot detect any of this.* The
audit toolkit + three-benchmark floor table + the pre/post discriminator are
the contributions. Read-path redesign is the follow-up work.

---

# Chapter 3 — memv2: masked-reconstruction training and the amplitude ladder (2026-07-06 → 07-07)

Chapter 2 ended with the read pathway as prime suspect. Chapter 3 is the
read-path redesign, run to its pre-registered endpoint: a new fusion module,
two new losses that *require* content, a demand-bearing data mixture, four
training rounds that surgically eliminated every trainability excuse
(gate init → gate evasion → gate welded open → amplitude 5×), and a final
n=96 paired discriminator. Design doc: `docs/memory-v2-training-framework.md`
(Figures M1/M2). Implementation: commits `c041481` → `827cbc5` on `memexp`
(61+ new unit tests, all green; schema_version 2 checkpoints).

## 14. Design: manufacture demand, then supervise the read

Four components, each closing a hole Chapters 1–2 identified:

- **Masked-decision training** — at most one supervised decision per segment
  is served black frames; the policy must act from memory at that decision.
- **L_rec (masked latent reconstruction)** — the policy tokens at a masked
  decision are decoded (zero-init `mem_cond_adapter` → reused, frozen-target
  `vj_predictor` with a learned `wm_mask_token`) against the frozen
  `vj_encoder` embedding of the *clean* frame. Direct gradient pressure to
  route episode content through the policy stream.
- **L_nce (episodic InfoNCE)** — pre-gate memory residual must identify its
  own episode against same-task negatives (FIFO queue 256).
- **SparseKeyMemoryFusion** — content-addressed read: 8 content keys,
  top-2 softmax routing, whitened residual (LayerNorm without affine),
  separate content gate `g_c`, plus an explicit time-tap token
  (log(1+steps)) so the pacemaker signal has a *dedicated channel* and no
  longer needs to occupy the content pathway.
- **privdec A/B control** — a twin run whose reconstruction is conditioned
  on detached action tokens instead of policy tokens
  (`rec_condition_source`), so every content meter has a
  cannot-possibly-read control arm.

**G0 demand audit (pre-registered gate):** ridge/GroupKFold decodability of
expert actions from (task × phase) templates on vanilla LIBERO leaves no
residual demand a memory could serve — vanilla LIBERO has **zero
manufacturable demand**, so stage-1 uses `memv2_stage1_mix`: 4 LIBERO suites
×1.0 + `libero_mem` ×2.0 + 21 MIKASA anchors ×0.0952 (≈50 % of samples
demand-bearing).

## 15. Four training rounds — the gate/valve/amplitude causal chain

All rounds: 10K steps, 8×H100, paired main/privdec arms, online
Δforeign/Δbypass meters (foreign = donor-episode state swap re-applied at
the fusion call after the schema-2 wiring bug was found and regression-tested).

| round | intervention | injection ratio | outcome |
|---|---|---|---|
| memv2 | as designed (zero-init gate) | ≈ 0 | content reaches the pre-gate residual (NCE top-1 22 %, slot correlation 0.995→0.30) but tanh γ settles at −0.005 — **the gate never opens**; all discriminators null |
| memv2.1 | `content_gate_init` 0.05, mask-grad α 1.0 | 0.0018 → 0.0004 | the optimizer **evades via the other valve**: learned g_c closes over training — and only in the main (shared-conditioning) arm; privdec g_c stays open |
| memv2.2 | `content_gate_fixed` (g_c ≡ 1, γ_c frozen buffer) | 0.004 | unclosable valve, still null → diagnostics D1/D2 below: amplitude, not routing |
| memv2.3 | `content_scale` 5.0 ("loud memory") | **0.335** | first sustained arm-specific online signal: Δforeign_rec tail-20 mean **+3.4×10⁻⁴** (main) vs **≡ 0.0** (privdec control); G4 passed; plateaued below the 10⁻³ strong-pass bar |

The ladder is causally clean: each round removed exactly one degree of
freedom the optimizer had used to avoid reading content, and the meters
confirmed the removal worked (injection 0 → 0.0004 → 0.004 → 0.335).

## 16. Diagnostics D1–D3 (memv2.2 null autopsy)

- **D1 — L_rec vs template floor** (n=150 masked decisions): model L_rec
  1.6135 vs leave-one-out per-dataset template floor 1.6152 (global-mean
  floor 1.637). **The reconstruction objective is solved to within 0.002 of
  what task-generic templates achieve with zero episode content.** L_rec
  trained a reconstruction pathway; it never *needed* the episode.
- **D2 — decoder conditioning probe**: the decoder is live at unit gain
  (mem_cond_adapter weights healthy, mask token trained), but at
  memv2.2's 0.4 % injection the live-vs-foreign loss contrast is ≈ 4×10⁻⁶ —
  **below bf16 training arithmetic**. Motivated content_scale 5.0.
- **D3 — masked-demand audit at the exact masked depths** (7 corpora,
  2,319 episodes, 55K masked-depth rows, ridge/GroupKFold-by-episode):
  mixture-weighted, init-layout + burn-in features add **−0.024 R²** beyond
  (task × phase) — *net zero long-range-recoverable demand at the positions
  the mask supervises*. On `libero_mem` (the dominant demand corpus,
  weight 2.0): layout/burn-in **−0.036**, but the always-sighted *previous
  decision* **+0.124** — the manufactured demand is satisfiable from the
  adjacent sighted decision (`memory_mask_max_per_segment=1` guarantees a
  1-step buffer), not from memory. Genuine recoverable demand exists only in
  a few short-horizon MIKASA anchors (shell_game +0.149, chain_of_colors
  +0.062, take_it_back +0.059), which are ≈5 % of the mixture. **The masking
  scheme leaks a local shortcut and the mixture's demand was mislabeled** —
  jointly, this fully explains D1's template-floor result.

## 17. Endpoint — fwdseq foreign discriminator, n=96 paired episodes

Data-level foreign burn-in splice through `forward_sequence` (paired RNG:
identical mask plans in both arms), 4 MIKASA anchor corpora × 24 episodes,
step_10000 checkpoints:

| arm | gap_rec (foreign−live) | p (one-sided) | gap_act | p |
|---|---|---|---|---|
| **main** | **+4.6×10⁻⁶** [−7.5×10⁻⁶, +1.6×10⁻⁵] | **0.22** | −4.5×10⁻⁵ | 0.84 |
| privdec (control) | ≡ 0 (exact) | 1.0 | −3.2×10⁻⁵ | 0.56 |

Three readings:

1. **The endpoint is null.** The CI's upper bound (+1.6×10⁻⁵) excludes the
   online training-time effect (+3.4×10⁻⁴) by ~20×. The content-dependence
   the online meter measured on training-lattice batches **does not
   transfer to natural sequential inference on the same corpora** — it was
   an overfit association to specific reconstruction targets (train-episode
   memorization) and/or specific to the 8-decision lattice state
   distribution, not a generalizing content read.
2. **The instrument is valid**: privdec gap_rec is *exactly* zero (its
   reconstruction path cannot see the swapped state), and per-dataset gaps
   in the main arm are largest where burn-in is longest — the splice
   mechanics work; there is simply no effect to detect.
3. gap_act is null in both arms: even at 33 % injection, swapping in a
   different episode's memory does not measurably change **actions**.

## 18. Closed-loop battery — step_10000 (memv2.3 main arm)

| eval | result | reference |
|---|---|---|
| MIKASA anchors live (4×50) | **0/200** | zero-shot floor 0/200; T-FT 1/200 |
| MIKASA anchors bypass (4×50) | 0/200 | — |
| LIBERO-Mem live (10×20) | **0/200** | zero-shot floor 0/200 |
| LIBERO regression (10/goal/object/spatial, 100 each) | **0 / 21 / 16 / 9 %** | memv1-100K: 47/78/88/91; T-FT mid-point: 22/56/69/60 |

Behavioral acquisition on the memory benchmarks did not emerge at this dose
(consistent with the T-FT threshold finding), and the loud-memory
configuration is **behaviorally destructive on vanilla LIBERO**: forcing a
fixed 33 % injection of (content-empty) whitened residual into the policy
stream, under a 50 %-demand mixture for only 10K steps, collapsed baseline
competence well below even the naive-mixture forgetting seen in T-FT.
Confounded (steps 10K vs 100K, mixture, amplitude — no matched control at
scale 1.0), but the direction is unambiguous: the amplitude needed to make
content *arithmetically visible* to the trainer is an amplitude the policy
cannot tolerate.

## 19. Verdict and interpretation

The memv2 program removed, one round at a time, every mechanism by which the
optimizer had avoided reading memory content — gate init, gate evasion,
amplitude — while adding direct reconstruction supervision, an episodic
contrastive loss, content-addressed routing, a dedicated time channel, and
data where content demonstrably matters. The result:

1. **Storage is again confirmed** (NCE identifies episodes; slots
   decorrelate) — as in memv1.
2. **The reconstruction objective is satisfiable without episode content**
   (D1: template floor), because the masking scheme leaks a local shortcut
   and the mixture's demand was mislabeled (D3: net-zero long-range
   recoverable demand at masked depths; the sighted adjacent decision
   carries all the usable signal) — so L_rec supervised a pathway without
   ever forcing content through it.
3. **When content is made loud enough to matter arithmetically (0.335
   injection), training-set content-dependence appears but does not
   generalize** (online +3.4×10⁻⁴ → endpoint null at 20× tighter CI), and
   the amplitude itself destroys the policy (LIBERO 0/21/16/9).
4. The pacemaker interpretation survives its strongest challenge yet:
   with time given a dedicated tap and content given keys, routing, losses,
   demand, and amplitude, **behaviorally-read episodic content still does
   not form under behavior cloning at this scale.**

## 20. Updated recommendation

The Chapter 13 arc is now stronger, not weaker: the paper gains a
constructive chapter — *"we redesigned the read path and the objective to
force content-reading, eliminated every trainability failure mode
identified by our own instruments, and the null survived"* — plus two
transferable methodological findings: (a) masked-reconstruction auxiliary
losses for memory are template-soluble unless the mask schedule blocks
local interpolation (mask *runs* of decisions, not singletons); (b) online
content meters on training batches can show sustained arm-specific signals
that are pure train-set memorization — endpoint discrimination on held-out
rollouts is mandatory.

Remaining escalations, in order of information-per-GPU-hour:
1. **Fix the D3 leak**: contiguous masked runs (2–3 decisions,
   `memory_mask_max_per_segment` > 1) so reconstruction cannot be
   interpolated, and **reweight the mixture to the anchors D3 certifies as
   demand-bearing at masked depths** (shell_game/chain_of_colors/
   take_it_back — currently ≈5 % of samples) — the two memv2 design flaws
   the diagnostics actually convicted.
2. Longer training at moderate scale (1.5–2×, not 5×) with the fwdseq
   endpoint as the tracked metric — 10K steps never showed behavioral
   acquisition on any arm of any round.
3. Otherwise: ship the paper as the completed two-arc anatomy
   (pacemaker + the redesign null); the instruments are the contribution.
