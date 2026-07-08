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
- **M1→M2 transition (10:35):** M1 verdict committed (29c790f, pushed).
  Both M2 arms launched: 6108357 (live) / 6108358 (priorread control),
  MICRO=1, SHA-pinned at 29c790f. Gate watcher v3 running from
  vlajepa_runs/m3_gate_watch/ (exports + fwdseq n=32 certified anchors at
  2.5K/5K/7.5K/10K + libero_goal guardrail on the live arm).
- **M2 @150 (10:50):** both arms stepping, correct identity, fresh start.
  retro_loss_raw 1.56/1.58 = M1 endpoint (1.587) → no BC erasure at start;
  action 0.19/0.24 falling; pick flowing. Gate watcher live.
- **M2 @1.6K (11:40):** action live 0.1516 vs priorread 0.1629 — live arm
  7% better (first reader-value hint, early/noisy). retro_raw 1.48 — BELOW
  the M1 endpoint (improving on robot data, zero erasure). pick_acc 0.87 on
  robot burn-in (vs 0.50 on video — robot episodes more discriminable).
  ~27 steps/min → 10K ≈ 6.2h (~17:50). 2.5K gate ~12:15.
- **2.5K gate FIX (13:15):** fwdseq disc assumed memv2 mask-plan machinery →
  schema-3 guard rejected it (both 2.5K gate jobs failed). Patched
  out-of-repo copy fwdseq_disc_m3.py: no injected mask plan for schema 3,
  rec := retro_loss (paired RNG keeps live/foreign mask runs identical),
  min_burn_in floored to 5. Watcher rewired + restarted; 2.5K gates
  resubmitted. Repo untouched (SHA-pin freeze); patch to be upstreamed into
  scripts/analysis/memv2_fwdseq_disc.py at the post-training commit window.
- **Guardrail deferred (13:20):** serve path crashes for schema 3 at
  server_policy.py:36 (unconditional policy_memory_fusion.float()). Tracked
  file + SHA-pin freeze → cannot fix mid-train. DECISION: drop the advisory
  mid-train LIBERO-goal guardrails (2.5K/5K); behavioral trend is covered by
  live-vs-priorread action loss; content gates by fwdseq (server-free). Fix
  + 2-episode serve smoke scheduled for the post-training window BEFORE the
  closed-loop endpoint battery.
- **2.5K GATES (13:50): PASS, decisively.** live gap_act = +2.25e-2
  (ci [+1.3e-2,+3.3e-2], p<0.001, n=24 held-out) — the action head reads
  episode-specific memory content; 200x above the pre-registered 10K PASS
  bar at 25% of training. priorread control gap_act = exactly 0 (p=1.0) —
  read severed => effect gone; instrument valid. REINTERPRETATION (logged):
  gap_rec is CONFOUNDED for schema 3 (the burn-in splice changes the retro
  targets themselves — both arms show ~+1.1e-2), so gap_act is the primary
  content-read endpoint; gap_rec demoted to a splice-sanity check. No knob
  changes — everything passing.
- **5K GATES + @7K TB (14:55): PASS, growing.** live gap_act +3.12e-2
  (p<0.001) — up from +2.25e-2 @2.5K; priorread still exactly 0. Action
  loss: live 0.074 vs priorread 0.105 — live arm 30% better (was 7% @1.6K):
  the reader's behavioral value grows with training. retro_raw 1.416, still
  improving (no erasure). Requeue boundary passed cleanly on both arms.
  10K ETA ~17:15.
- **7.5K GATES (15:58): PASS, still growing.** live gap_act +4.94e-2
  (trajectory 2.25 → 3.12 → 4.94 e-2 across gates); priorread exact 0
  throughout. Action: live 0.0766 vs priorread 0.0849 @8.7K. ~1.3K steps
  to 10K (~16:45), endpoint battery follows the post-training window.
- **M2 COMPLETE (16:50), post-training window:** server schema-3 None-guard
  fixed (server_policy.py); fwdseq schema-3 patch upstreamed; report M2
  section written. Both 10K ckpts exported by the watcher.
