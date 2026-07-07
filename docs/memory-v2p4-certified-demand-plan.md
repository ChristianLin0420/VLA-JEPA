# memv2.4 — Certified Demand: Plan and Pre-registration

**Date:** 2026-07-07 · **Branch:** `memexp` · **Predecessor:** memv2 program
(Chapter 3 of `docs/memory-experiments-report.md`, endpoint null at commit
`e558962`).

memv2.4 is one round, not a new program. It fixes exactly the two design
flaws the memv2 diagnostics convicted, adds the one arithmetic fix D2
motivated, and pre-registers a decision rule that makes the round
informative in both directions. Nothing else changes: same warm start, same
architecture (SparseKeyMemoryFusion, schema 2), same losses, same 10K-step
budget, same paired main/privdec control arms.

---

## 1. The three convicted issues and their fixes

| # | Issue (evidence) | Fix | Where |
|---|---|---|---|
| 1 | **Interpolation shortcut.** `memory_mask_max_per_segment=1` guarantees every masked decision a sighted immediate neighbor; D3: on `libero_mem` the sighted previous decision explains +0.124 R² vs −0.036 for everything long-range. L_rec was solvable by local interpolation (D1: template floor). | Mask **contiguous runs of 2 decisions**: the deeper position of the run has no sighted neighbor at stride distance. | `_sample_mask_plan` (`memory_mask_run_len`) |
| 2 | **Mislabeled demand.** Mixture-weighted long-range-recoverable demand at masked depths = −0.024 R² (net zero). `libero_mem` (weight 2.0) carries none; only short-horizon MIKASA anchors are real (shell_game +0.149) at ~5 % of samples. | **Certify before training** (Phase 0): D3 audit over all 21 anchors; build `memv2_stage2_mix` only from corpora with gap ≥ +0.04; gate C0 on the weighted total. | `d3_masked_demand_audit.py` → `mixtures.py` |
| 3 | **Amplitude dilemma.** Scale 1.0 → 4×10⁻⁶ contrast, below bf16 training arithmetic (D2); scale 5.0 → injection 0.335 destroys the policy (LIBERO 0/21/16/9). | **fp32 reconstruction branch** (`rec_loss_fp32`) lowers the arithmetic floor ~250×. **D2′ result (2026-07-07): at scale 1.0 in fp32 the contrast is +2.1×10⁻⁵, p=0.034, all 24/24 pairs resolved — trainable at the ORIGINAL amplitude.** Off-trained-point scale overrides (2.0/3.0) add nothing, so memv2.4 reverts the loud-memory hack entirely: `content_scale: 1.0` + fp32. The policy stream is left undisturbed (injection back to ~0.4 %), protecting LIBERO competence at the source. | `_compute_recon_loss` + config |

Process fix (memv2.3's lesson): the **online Δforeign meter is demoted to
telemetry**. The tracked go/no-go metric is the fwdseq foreign discriminator
on held-out rollouts, run at every 2.5K-step checkpoint (n=32), alongside a
LIBERO-goal mini-regression (20 eps) as a competence guardrail.

## 2. Flow

```
PHASE 0 · CERTIFY (CPU-class job, no training until demand is proven)
──────────────────────────────────────────────────────────────────────
  d3_masked_demand_audit.py on all 21 MIKASA anchors + libero_mem
  (7 corpora already done in the reduced run; 15 submitted now)
            │
            ▼
  per-corpus demand gap  =  R²(task,phase,layout,burn-in) − R²(task,phase)
            │
            ▼
  ┌─ GATE C0 ─────────────────────────────────────────────┐
  │ certified set = {corpus : gap ≥ +0.04}                │
  │ PASS if the buildable mixture reaches weighted demand │
  │ ≥ +0.05 R² over its demand-bearing half               │
  └──────┬────────────────────────────────┬───────────────┘
     PASS│                            FAIL│
         ▼                                ▼
  memv2_stage2_mix                 STOP: no trainable demand in
  · certified anchors upweighted   our data; ship the paper; the
  · uncertified anchors dropped    lever is a new benchmark, not
  · libero_mem demoted to ×1.0     a new training trick
  · 4 vanilla LIBERO kept ×1.0
    (competence anchor)

PHASE 1 · IMPLEMENT (small diffs, all unit-tested)
──────────────────────────────────────────────────────────────────────
  a) memory_mask_run_len=2 — contiguous-run mask planner
  b) rec_loss_fp32 — fp32 decoder branch for L_rec
  c) D2′ probe: rec contrast at content_scale {1,2,3} in fp32 on the
     memv2.2 ckpt (n=24) → confirm scale 2.0 clears the fp32 floor
  d) gate watcher: at ckpt steps {2500,5000,7500,10000} export +
     submit fwdseq n=32 + LIBERO-goal 20 eps

PHASE 2 · TRAIN (two parallel 8×H100 jobs, 10K steps, ~8 h)
──────────────────────────────────────────────────────────────────────
     main arm                         privdec control arm
        │                                    │
        ├── @2.5K  GATE G1: fwdseq gap_rec > 0 trend
        │          AND LIBERO-goal ≥ 50 % — else KILL EARLY
        ├── @5K    GATE G2: gap_rec growing, control ≡ 0
        └── @10K   export (schema-2), full endpoint battery

PHASE 3 · VERDICT (all evals in parallel, ~2 h)
──────────────────────────────────────────────────────────────────────
  fwdseq n=96 both arms · MIKASA live/bypass/foreign ·
  LIBERO-Mem · LIBERO full regression
            │
            ▼
  ┌─ PRE-REGISTERED DECISION RULE ──────────────────────────────┐
  │ PASS: gap_rec ≥ 1×10⁻⁴ one-sided p<0.01, privdec ≡ 0,      │
  │       LIBERO-goal ≥ 60 % → content read exists → scale up   │
  │ FAIL: null again → strongest possible negative (demand      │
  │       certified, shortcut closed, arithmetic visible,       │
  │       amplitude tolerable) → ship the two-arc paper         │
  └──────────────────────────────────────────────────────────────┘
```

## 3. Config diff vs memv2.3 (both arms)

```yaml
datasets.vla_data:
  data_mix: memv2_stage2_mix          # was memv2_stage1_mix
  memory_mask_rate: 0.25              # unchanged
  memory_mask_max_per_segment: 2      # was 1 (bookkeeping cap = run len)
  memory_mask_run_len: 2              # NEW — contiguous runs
framework.memory.action_conditioning:
  content_scale: 1.0                  # was 5.0 — loud-memory hack reverted
                                      # (D2′: contrast fp32-trainable at 1.0)
  content_gate_fixed: true            # unchanged (2.1 lesson)
  content_gate_init: 1.0              # unchanged
trainer:
  rec_loss_fp32: true                 # NEW — fp32 L_rec branch
```

Everything else — warm start (`vlajepa_memv1_video` final), losses
(rec 0.5 / nce 0.2), `mask_grad_alpha` 1.0, BPTT 4, burn-in 8, 10K steps —
is held fixed for comparability with rounds 2 → 2.3.

## 4. Why run-length 2 (not 3)

With K=4 supervised decisions and the first never masked, a run of 3 leaves
exactly one sighted decision per masked segment — maximal demand but also
maximal action-BC starvation on demand-bearing corpora (25 % of segments
would supervise 3 of 4 decisions blind). A run of 2 already breaks the
1-step interpolation shortcut at its second position while keeping half the
supervised window sighted. Run-length 3 is the pre-registered escalation if
memv2.4 shows a positive-but-weak endpoint.

## 5. Cost and abort points

- Phase 0: one CPU-class Slurm job (~2 h), zero GPU-training risk.
- Phase 1: half a day of implementation + a 15-min GPU probe.
- Phase 2: 2 × 10K steps ≈ 2 × 8 h on 8×H100 — the only real spend, and
  G1 can kill it at 25 % if either the discriminator stays flat-negative or
  competence collapses again.
- Phase 3: ~2 GPU-hours of evals.

Total worst case ≈ one memv2 round; best abort case ≈ 2 h of CPU time.
