# memv3 Retro-JEPA — Hyperparameter & Design Decision Log

Autonomous-run log (2026-07-08 →). Every default chosen, every deviation from
the design doc, and every tuning decision goes here with rationale. Lives
outside the repo while SHA-pinned jobs run (untracked files trip the requeue
guard); moved to `docs/memory-v3-decision-log.md` and committed at each stage
boundary. Tuning policy at the bottom.

## Stage M1 decisions (set before launch, 2026-07-08 ~05:40)

| knob | value | rationale |
|---|---|---|
| sequence length | 16 frames → 8 latent frames | one encoder pass per sample; 8 memory steps; ≥5 needed by the run sampler |
| SSv2 frame stride | 2 | ~2.7 s span; stride 1 spans too little time for memory demand |
| robot-farm frame stride | 4 | compromise between LIBERO (fps 20, decision stride 7) and MIKASA (fps 10); span 61 frames fits most episodes |
| mask run length k | U{3..6} | k≥3 guarantees ≥1 interior memory-only target; k≤6 keeps ≥1 past frame visible |
| frame 0 | never masked | anchor context, mirrors memv2's first-decision-sighted invariant |
| L_pick temperature | 0.07 | CPC/MoCo convention |
| L_pick negatives | all pooled frames of the batch (B·T=32) | same-episode + cross-episode; no FIFO queue in M1 (batch supplies enough; one less mechanism) |
| λ_retro / λ_pick (M1) | 1.0 / 0.2 | retrodiction is the primary writer objective; pick is the anti-template regularizer |
| batch | 4/GPU × 8 GPUs | 16-frame encodes; measured throughput ~112 steps/min — could have been larger, not worth a restart |
| steps | 30,000 (~4.5 h) | ≈4.4 epochs of SSv2-equivalent draws; gates decide sufficiency, not the step count |
| LRs | base 1e-4, vj_predictor 3e-5, memory 1e-4 | predictor is warm (video-pretrain final), writer/heads are cold |
| warm start | `vlajepa_memv1_video` final | predictor already knows forward prediction; retrodiction is grafted onto it (shared-blocks design) |
| frozen | Qwen, vj_encoder, action head | M1 never calls Qwen or the action head; encoder is the frozen teacher |
| writer pooling | parameter-free group-mean 256→8 tokens | no learnable writer-input bottleneck to erase; strict divisibility check |
| M1 corpus | SSv2 (220,847) + robot farm (6,154 eps, main camera) | proportional mixing ⇒ robot ≈ 3% of draws; acceptable because M2 continues retro on robot data; revisit only if M2 gates fail |

## Deviations from the design doc (disclosed)

1. **Dropped the 1K writer-freeze warmup in M2.** The M1-pretrained writer
   already hands the reader a contentful state at step 0, and the retro loss
   keeps training it; one less schedule mechanism. Revisit if the M2 gate
   ladder shows early reader collapse.
2. **Control arm = prior-read, not shuffled-state.** per_device_batch=1 makes
   batch-shuffling degenerate; the control reader receives the learned
   initial state's read tokens instead (content- and maturity-empty). The
   decisive comparisons: live-arm fwdseq gap_rec (n=96) and live-vs-priorread
   action loss / closed-loop success.
3. **Serve-time step latents = current frame duplicated to one tubelet**
   (training uses the decision clip's final latent frame). Static-motion
   approximation at closed-loop time; fwdseq endpoints are unaffected
   (training-faithful path). Flagged for the report's caveats.
4. **No online Δforeign/Δbypass meters in M2** — the memv2.3 lesson (online
   meters can show pure memorization) demoted them; the fwdseq gate ladder is
   the tracked instrument.

## Observations & mid-run notes

- **M1 @2K:** pick_acc 0.50 (chance 0.03), prior_gap +0.017, retro 1.63.
- **M1 @6K:** pick_acc plateaued at 0.50; prior_gap doubled to +0.034;
  retro 1.60. Interpretation: adjacent frames inside a masked run are
  genuinely hard to tell apart (contiguous-run ambiguity), so 0.5 may be
  near the task ceiling rather than a defect; prior_gap growth is the gate
  that matters and it is healthy. **Decision: no mid-run change.** If M2's
  pick_acc sits at chance on robot data, raise λ_pick to 0.5 at a stage
  boundary (logged here first).

## Tuning policy (standing)

- **No mid-run hyperparameter changes.** Changes only at stage boundaries or
  after a pre-registered gate fails, each logged here with before/after and
  rationale. A failed gate triggers: diagnose from telemetry → single-knob
  change → relaunch → log.
- **Commit freeze:** never commit or create repo files while an
  EXPECTED_GIT_SHA-pinned run is live; commits happen in the transition
  windows between stages.
- **M2 λ schedule (pre-decided):** start λ_retro 0.5 / λ_pick 0.2. If
  guardrail (LIBERO-goal) falls while retro improves → halve λ_retro. If
  retro_loss rises >5% from its M1 endpoint (BC erasure) → double λ_retro.
  One change per gate window, maximum.

- **M1 @14.4K (07:55):** prior_gap +0.054 (steady growth: .017→.034→.054);
  pick_acc still 0.50 (ceiling interpretation holds); retro 1.592. All gates
  healthy; no action. One requeue boundary expected ~09:40 before the
  ~30K finish (~10:15).
- **M1 @21.6K (08:57):** prior_gap +0.069 (monotone: .017/.034/.054/.069);
  retro 1.586; pick_acc 0.50. Healthy. ~8.4K steps left; requeue boundary
  ~09:40 then finish ~10:00.
- **M1 @27.9K (10:00):** requeue boundary passed cleanly (segment 2 running).
  prior_gap +0.089 — monotone through the entire run; retro 1.578;
  pick 0.139/acc 0.50. ~18 min to completion.
- **M1 COMPLETE @30K (10:27): VERDICT PASS.** Final prior_gap +0.096 —
  monotone the whole run (.017/.034/.054/.069/.089/.096), never plateaued:
  the memory's episode-content contribution to retrodiction kept growing to
  the last step. pick_acc 0.50 (chance 0.031); retro 1.84→1.587;
  pick 0.556→0.139. Wall time ~4.5h across 2 requeue segments. → launch M2.
