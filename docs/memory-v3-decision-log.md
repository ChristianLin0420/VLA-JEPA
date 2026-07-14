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
- **10K ENDPOINT (17:40): live gap_act = +7.03e-2 [CI +5.7,+8.5 e-2],
  p<0.0001, n=72 held-out; priorread exactly 0 at n=72.** Full trajectory
  2.25/3.12/4.94/5.16(n=24)/7.03(n=72) e-2 — 700x the pre-registered PASS
  bar. Smoke found a server dtype bug from my own fix (fp32 projs vs bf16
  model) — projs now follow model dtype (f20b23b); smoke resubmitted; the
  first smoke's EGL teardown error is watched as possibly secondary.
- **Closed-loop battery launched (18:20):** serve smoke round 2 proved the
  schema-3 server (spatial ran 20 full episodes; 3 suites hit node EGL
  flake — transient, full runs will confirm). Jobs: MIKASA 6114207-210
  (live/bypass x live/priorread ckpts, 50 trials), LIBERO-Mem 6114211-212
  (both ckpts, 20 trials), LIBERO regression 6114213 (live, 4 suites x 20).
- **MIKASA closed-loop (19:25): 0/200 both ckpts (live and priorread; both
  memory modes).** At floor, as the pre-registered caveat anticipated —
  10K steps has never produced MIKASA behavioral acquisition in any round
  (T-FT 1/200 at 15K, memv2.x 0/200). The content read is established by
  the discriminator; converting it to MIKASA task success is a dose
  question (longer M2 = the recommendation). LIBERO-Mem + regression pending.
- **Serve-time collapse detected (19:55):** LIBERO regression emerging at
  ~0-1% despite the live arm's best-ever action loss. Hypothesis: the
  disclosed static-frame serve approximation is now HARMFUL because the
  policy genuinely reads the memory (gap_act +0.07) — OOD writer input =>
  corrupted read tokens => corrupted actions; the stronger the read, the
  worse OOD memory hurts (memv1/2 were immune because they ignored it).
  Probe submitted: live ckpt, MEMORY_MODE=bypass, 5 trials — if success
  jumps, diagnosis confirmed => implement serve-time frame buffer (real
  clips to the writer, train/serve parity), re-run closed-loop.
- **Diagnosis pivot (20:45):** bypass probe ALSO at floor (0/50 per suite),
  priorread ckpt (mismatch-free by construction) also 0/200 on libero_mem.
  The serve static-frame theory cannot be the sole cause; common factor =
  DOSE: 10K from a video warm start on 34%-vanilla stage2 mix is below
  closed-loop acquisition for every arm (memv2.4 guardrail matched this on
  the same mix; memv1 needed 30K+). Teacher-forced action losses are healthy.
  DECISION (within granted tuning authority, stage boundary): EXTEND M2 to
  40K steps, both arms, same mixture (mixture change would break arm
  comparability), gates at 20K/30K/40K; implement the serve-time frame
  buffer now regardless (parity for the 40K battery); no other knob moves.
- **M2 extended to 40K (21:25):** both arms resumed from step_10000 full
  state (jobs 6116263/64, MAX_TRAIN_STEPS=40000, SHA a27f6e3). Frame-buffer
  serve fix committed (rolling 8-frame per-view history -> real writer
  clips); the failing unit test was chased to a stub degeneracy (scale-only
  latents die in the writer LayerNorm) — production latents unaffected; 14
  tests green. Watcher rescheduled to 20K/30K/40K gates with functional
  guardrails (fixed server). Same mixture, no other knob moves.
- **40K extension @11.7K (22:20):** both arms resumed from step_10000
  cleanly. Action 0.108/0.111 — transient bump above the 10K tail
  (0.077/0.085), consistent with warm-restart scheduler; watch it recover
  by ~14K. Live-vs-priorread ordering preserved.
- **@13.5K (23:22):** transient recovered — live action 0.0687 (new best),
  priorread 0.0987 (live +31%); retro_raw 1.398 still improving. Healthy.
- **@15.3K (00:25):** live 0.0612 / priorread 0.0816 (+33% live advantage,
  widening). Requeue boundary ~01:20; 20K gate ~03:00.
- **Requeue failure + fix (01:30):** stale .training_complete from the 10K
  run short-circuited the boundary requeue ("training complete -> no
  requeue") — both arms stopped at 16,567 with clean boundary saves.
  Markers deleted, arms resubmitted (resume from step_16567). RULE for any
  future extension: clear .training_complete before relaunching.
- **Resume verified (02:00):** 6118142/43 resumed from step_16567/16452,
  stepping. ~23.4K steps remain ≈ 14h ≈ 4 requeue boundaries (markers clear).
- **@19K (03:00):** live 0.0670 / priorread 0.1033 (+35% live advantage,
  still widening). 20K gate fires within ~40 min.
- **20K gate (04:05): live gap_act +7.84e-2 (p<0.001)** — trajectory
  continues 7.03 -> 7.84 across the extension. Priorread gate + 20K
  guardrail (6119478) in flight. Watcher guardrail steps were still
  2500/5000 — fixed to 20K/30K/40K and watcher restarted cleanly.
- **20K gates complete (05:05):** priorread gap_act exact 0 ✓ (control valid
  through the extension). Guardrail at 153/200 eps: 2 successes (~1.3%) —
  first nonzero closed-loop of the program but marginal; flagged per policy,
  continue to 30K/40K and judge the SLOPE (memv1 needed 30K+ for LIBERO
  competence from a stronger warm start).
- **@24.2K (06:08):** requeue boundary passed cleanly (marker fix works).
  Final 20K guardrail: 1.5% (3/200). Action live 0.0446 / prior 0.0659
  (+32%), both steadily improving. 30K gate ~08:30.
- **@25.9K (07:10):** live 0.0521 / priorread 0.0871 (+40% live advantage —
  the widest yet). 30K gate ~08:40.
- **Post-dark-period collection (07-11):** wakeup chain was dark ~2 days
  (harness gap); both arms COMPLETED 40K on 07-09 15:45, watcher did all
  exports + 30K/40K gates before going idle. Collected: gap_act trajectory
  2.25/3.12/4.94/7.03/7.84/10.58/11.19 e-2 (2.5K->40K, ~5x growth, control
  exact 0 at all 7 gates). Guardrail: 1.5/1.5/2.0% at 20/30/40K — FLAT:
  the dose extension did NOT convert to LIBERO closed-loop competence; the
  cap is the mixture/warm-start (34% vanilla from video start), not dose.
  40K endpoint battery launched (6139039-47: n=96 both arms, MIKASA
  live+bypass both ckpts, LIBERO-Mem both, lbreg live).
- **40K endpoint collected (07-12 00:40):** live gap_act +1.427e-1
  [CI +1.19,+1.68 e-1] p<1e-4 n=72; priorread exact 0 at n=72. Trajectory
  final: 2.25 -> 14.27 e-2 (6x growth, 9 measurements, control 0 at all).
  MIKASA 40K: floor (0-0.5%) both ckpts both modes. LIBERO-Mem running,
  lbreg pending (weekend queue backlog).
- **Mixture-semantics correction (07-12):** get_vla_dataset builds with
  balance_dataset_weights=False → per-draw share = weight/sum(weights),
  SIZE-INDEPENDENT. memv2_stage2_mix therefore actually drew ~35% vanilla /
  4% libero_mem / 61% anchors (not the frames-weighted 34/20/47 documented
  earlier — to be corrected in the report). Strengthens the competence-cap
  diagnosis. memv3p1_mix calibrated under the CORRECT semantics: 67% vanilla
  / 4% libero_mem / 29% certified anchors per draw.
- **memv3.1 launched-prep (07-12):** warm-start surgery = allv2-100K
  backbone+action head (LIBERO 47/78/88/91) + m3-live-40K memory stack incl.
  retro predictor (merge job 6139670). Gentle LRs to protect competence;
  retro losses on; gates must show BOTH gap_act >> 0 (read preserved) AND
  guardrail >= ~50% (competence preserved).
- **FINAL (07-12 02:40):** LIBERO-Mem 0/200 live, 0/190 priorread
  (time-limit truncation). Report finalized: split verdict (read PASS
  1400x / conversion not achieved), mixture correction, three-era Program
  verdict, lessons, m3.1 recommendation. memv3 CLOSED; m3.1 launching.
- **m3.1 step-0 parallel validation (03:05):** while 6139847 queues, the
  MERGED checkpoint is being evaluated directly — fwdseq n=32 (read intact
  at step 0?) + LIBERO 4-suite (competence inherited at step 0?). Baselines
  the graft premise before training; also gives the dual-gate references.
- **m3.1 step-0 fwdseq (04:15): gap_act = -0.0013 (p=0.74) — the unadapted
  allv2 head does NOT yet read the grafted memory** (expected: it has never
  seen read tokens). Baseline interpretation: the graft's 10K must re-form
  the read into this head — with content already present in the writer, the
  ladder should show it forming much faster than memv3's from-scratch curve
  (which needed the full run to reach +0.14). Training + step0-lbreg still
  queued.
- **(05:15)** m3.1 training + step0-lbreg still PENDING ~2h (weekend queue
  congestion). Nothing actionable; chain holds.
- **m3.1 STARTED (07:20), step 60:** merged ckpt loaded strict. Memory
  transplant VERIFIED: retro_raw 1.309 (below the m3 endpoint), pick_acc
  1.0 on robot burn-ins — the grafted stack reads perfectly. Action loss
  0.23-0.33 — elevated vs the competent-head hope (~0.05-0.1): the allv2
  head is disturbed by the 8 unadapted read tokens (consistent with step-0
  gap_act ~0). This is what the 10K adaptation is for; decision point =
  2.5K dual gate (action should be << 0.1 by then, read re-forming).
- **m3.1 @1.5K (08:20): adaptation curve excellent** — action loss quarters
  0.108/0.103/0.060/0.047: already BELOW memv3's 40K endpoint (0.052) at
  1.5K steps. retro_raw stable 1.34 (zero erasure). The competent head is
  absorbing the read tokens fast. 2.5K dual gate ~35 min.
- **m3.1 watcher bug (08:25):** single-arm loop hit set -u (ARMS[1] unbound)
  after the 2500 fwdseq submit — guardrail missed. Loop fixed, watcher
  restarted; 2500 guardrail + parallel MIKASA-25 submitted manually.
- **m3.1 2.5K gate (09:10): READ GATE GREEN** — gap_act +2.74e-2 (p<0.001),
  re-formed from ~0 at merge to above memv3's from-scratch 2.5K pace. The
  graft carries content that a NEW head learns to read in 2.5K steps.
  MIKASA-25: 1/100 (floor at this depth, expected). Guardrail (competence)
  still running — the decisive half.
- **2.5K guardrail EGL-flaked (09:40):** 1 episode then client stall (server
  clean) — the recurring sim-side flake; retried on a fresh node. Training
  already at 5.1K, action 0.046 (still descending). Read gate remains green.
- **DUAL GATES GREEN (11:45) — PROGRAM GOAL STATE REACHED.** 2.5K guardrail
  retry: 88% LIBERO-goal (200 eps) — ABOVE the allv2 no-memory reference
  (78%) at just 2.5K adaptation steps. 5K read gate: gap_act +3.61e-2
  (growing: 2.74 -> 3.61). The graft recipe delivers the first competent
  AND reading checkpoint of the program. Riding the ladder to 10K, then the
  full battery decides the final m3.1 verdict.
- **@7.5K gate (12:47): gap_act +3.85e-2** (2.74/3.61/3.85 — growing,
  saturating gently). MIKASA-25 on 5K ckpt submitted (6145960). 5K guardrail
  queued behind. 10K ETA ~13:30.
- **m3.1 COMPLETE @10K (13:40):** full parallel battery launched (endpoint
  n=96, LIBERO 4-suite, LIBERO-Mem, MIKASA live+bypass). 5K guardrail
  EGL-flaked empty (superseded by the battery's regression); 5K MIKASA-25
  floor as expected.
- **m3.1 10K endpoint (14:42): gap_act +6.20e-2 (n=72, p<1e-4)** — read
  ladder 2.74/3.61/3.85/6.20 e-2, still growing at 10K in a COMPETENT
  model. MIKASA 0-1% (floor, honest note: memory->MIKASA conversion
  remains future work). LIBERO-Mem at 104/200; 4-suite regression queued.
- **m3.1 FINAL (07-12 15:30): DUAL-GATE PASS — program complete.** LIBERO
  66.5/91.5/96/94.5 — ABOVE the no-memory baseline (47/78/88/91) on every
  suite, with read endpoint +6.2e-2 (p<1e-4) still growing. LIBERO-Mem +
  MIKASA remain floor (read->memory-task-success conversion = future work,
  now startable from a competent+reading checkpoint). Final model:
  vlajepa_runs/vlajepa_m3p1_graft/checkpoints/VLA-JEPA-m3p1-graft-step_10000.pt
- **m3.2 memory-dose LAUNCHED (07-12 16:20, job 6146759, SHA c4c02a7):**
  from the m3.1 ckpt; memv3p2_mix (libero_mem 31.6% / six strongest anchors
  47.4% / vanilla 21.1% per-draw); action LR restored 1e-4; 20K steps;
  MIKASA eval unnorm fixed to mikasa_robo. Gates 5K/10K/15K/20K = fwdseq +
  goal-retention guardrail (>=80% of 91.5%) + libero_mem emergence eval.
  Diagnosis basis: LIBERO-Mem freeze-at-subgoal (103-decision timeouts,
  video-confirmed mid-air stall), MIKASA open-loop under-fit (act loss
  3-6x LIBERO).
- **m3.2 started (19:02), step 590:** m3.1 ckpt loaded strict, fresh run
  dir, zero Tracebacks. Action 0.0508 — starts LOW as expected (competence
  carried over even under the libero_mem-heavy mixture); retro_raw 1.307 —
  read machinery intact. Healthy. 20K ETA ~7h; first gate at 5K (~1.7h).
- **m3.2 5K fwdseq (22:38): gap_act +4.0e-2 (p=0.003)** — read alive under
  the inverted mixture. Guardrail + libero_mem emergence evals queued.
- **m3.2 watcher tag bug (23:42):** guardrail block gated on tag="live" but
  m3.2's tag is "memdose" — 5K guardrail+libero_mem never submitted. Fixed,
  watcher restarted, both 5K evals submitted manually. Training requeued at
  its 4h boundary (marker clean, resumes ~5.4K).
- **m3.2 5K verdicts (04:55): retention GREEN — 82.5% goal** (bar ~73%);
  the inverted mixture is not costing competence. libero_mem at 79/200 eps:
  0 successes, 0 early terminations — freeze not yet broken at 5K (only
  ~1.6K libero_mem-equivalent steps); the 10K wave carries the pre-logged
  tuning decision. Read gate was already green (+4.0e-2).
- **m3.2 ESCALATION FIRED (07-13 07:00):** 10K gate 0-success/0-early-term
  at 112 eps + retention healthy -> pre-logged single knob: libero_mem
  weight 6.0->10.0 (share 31.6%->43.5%). Killed 6146759 (boundary saves
  intact), committed fc2b73c, relaunched as 6153703 (resumes ~11K, runs to
  20K under the heavier mixture). EXPECTATION SET: if the remaining ~9K at
  43.5% share also yields 0/0, the freeze is NOT dose-limited -> report
  verdict pivots to structural BC limitation on repetition subgoals; next
  lever = RL or transition-biased sampling.
- **10K libero_mem baseline FINAL (08:30): 200 eps, 0 successes, 0 early
  terminations** — the pre-escalation baseline is locked. Training +
  10K guardrail still queued.
- **Escalated run live (10:05):** resumed at 11,500 under 43.5% libero_mem.
  **10K guardrail: 90% goal** — retention rock solid under the memory-heavy
  mixture (m3.1 ref 91.5%). All gates green except the emergence question,
  which the 15K/20K waves decide.
- **15K verdicts (12:32): retention 95% (!!) — climbing under the 43.5%
  mixture (82.5/90/95). libero_mem STILL 0/0 at 150 eps after ~4K escalated
  steps. The structural interpretation hardens: read alive + retention
  excellent + heavy in-distribution dose + zero early terminations = BC
  does not produce repetition-subgoal behavior on this data. 20K finalizes.
- **m3.2 FINAL (07-13 16:30): STRUCTURAL VERDICT.** 20K: endpoint gap_act
  +0.118 (n=72, p<1e-4, program max); retention ladder 82.5/90/95/90.5;
  lbreg 55.5/90.5/92/96; libero_mem 0/200 with ZERO early terminations at
  every gate (800 episodes total). BC does not produce repetition-subgoal
  selection regardless of dose — expert multimodality at freeze states
  averages to hover. MIKASA unnorm saga: franka (program-long, ~6% distort)
  -> mikasa_robo (nonexistent key, jobs failed) -> new_embodiment (correct;
  rerun 6160975/76 in flight = first properly-unnormalized MIKASA numbers).
  Next variant: transition-biased sampling (cheap) or RL sparse-success
  (principled).
