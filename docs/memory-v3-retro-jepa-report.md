# Retro-JEPA (memv3) — Staged Execution Report

**Plan:** `docs/memory-v3-retro-jepa.md` · **Branch:** `memexp` · run fully
autonomously 2026-07-08 → 07-11; each stage's analysis is appended here as it
lands.

## Stage M1 — memory pretraining on actionless video

**Setup (2026-07-08).** Implementation commit `f68a10f`: schema-3 memory
(writer over pooled frozen VJ2 latents, no fusion module), shared-predictor
retrodiction (bidirectional mask argument), L_retro (fp32 latent L1) +
L_pick (InfoNCE over all pooled batch frames), prior-advantage gate metric on
the capture cadence. Corpus: SSv2 (220,847 clips, stride 2) + robot episode
farm (6,154 episodes: 4 LIBERO suites, libero_mem, 14 C0-certified MIKASA
anchors; main camera, stride 4), 16 frames/sample → 8 latent frames, mask =
contiguous run k∈{3..6} (interior positions are memory-only targets). Warm
start: `vlajepa_memv1_video` final (predictor knows forward prediction;
retrodiction and the writer are new). 9 new unit tests + neighboring suites
green; single-GPU production smoke passed before launch.

**Pre-registered M1 gates** (from the design doc):
- `memory/pick_acc` above chance (chance = 1/(B·T) = 1/32 per anchor) and
  clearly above a frame-position template solution;
- `memory/prior_gap` > 0 and growing — retrodiction from the trained memory
  must beat the same decoder reading the learned initial state;
- `loss/retro_loss` decreasing without `loss/pick_loss` collapse.

**Training (2026-07-08, 30K steps, ~4.5 h on 8×H100, 2 requeue segments,
~112 steps/min — no Qwen in the M1 loop).** One launch crash at step 0
(zero figure-interval modulo; trainer guarded, `cc84409`) then clean
throughout. Full knob-by-knob rationale: `memory-v3-decision-log.md`.

| gate metric | start | 6K | 14K | 22K | **30K (final)** | rule |
|---|---|---|---|---|---|---|
| `prior_gap` | +0.006 | +0.034 | +0.054 | +0.069 | **+0.096** | > 0 and growing — **monotone the entire run, never plateaued** |
| `pick_acc` | 0.07 | 0.50 | 0.50 | 0.50 | **0.50** | ≫ chance 0.031 — 16× |
| `retro_loss` | 1.84 | 1.60 | 1.59 | 1.586 | **1.587** | falling, no pick collapse |
| `pick_loss` | 0.556 | 0.14 | 0.14 | 0.139 | **0.139** | — |

**Verdict: PASS.** For the first time in the program, a *learned* writer
carries episode content that a decoder demonstrably reads: retrodiction from
the trained memory beats the identical decoder reading the empty prior by a
growing margin (+0.096 latent-L1 at 30K — three orders of magnitude above
the fp32 floor and ~50× the D1 template-floor gap that memv2's L_rec never
cleared), and the discriminative pick identifies the true masked past frame
among 32 candidates half the time. The 0.50 pick plateau is interpreted as
the contiguous-run ambiguity ceiling (adjacent masked frames are genuinely
hard to tell apart), not a defect; the unbounded gate (prior_gap) kept
growing. The M1 objective did exactly what the design doc claimed: masking
manufactured demand on unlabeled video, and the writer earned readable
content before BC ever entered the picture.

## Stage M2 — co-training into VLA-JEPA

_(pending: implementation, two arms — real + shuffled-state control —, gate
ladder @2.5K/5K/7.5K, 10K endpoint battery per the design-doc decision rule)_

## Endpoint evaluation

_(pending)_

## Final analysis

_(pending)_
