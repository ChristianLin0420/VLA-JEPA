# memv1 findings at cotrain step 34729 (2026-07-03, overnight Tier-0 campaign)

All closed-loop cells: 100 paired episodes/suite (10 trials × 10 tasks), seed 7,
`WITH_STATE=false`, identical weights (`ckpt_sha 2b981faac806`), serve-time mode only.
Full records: wandb `crlc112358/vla-jepa`, group `step_34729`; raw JSONL under
`results/<suite>/VLA-JEPA-memv1-<arm>-step_34729/`.

## 1. Headline triad (T0.1)

| pooled /400 (4 suites) | live | prior | bypass |
|---|---|---|---|
| success | **148** | ~17 | 46 |

live−bypass and live−prior both p < 1e-18 (McNemar, pooled goal+object+spatial).
Measured **injection ratio ≈ 0.45** (not the 0.022 the raw gate implies — the
residual norm is ~20× the consumer norm). Eval stack is fully deterministic:
5 identical live repeats, episode-for-episode (replicate SD = 0; seeds are the
replication axis; capture-context must be held fixed).

## 2. Content vs statistics (T0.4 transplant grid)

| arm /400 | success | vs live (McNemar) |
|---|---|---|
| noreset (stale cross-episode) | 158 | n.s. |
| noisematch (static moment-matched Gaussian) | 152 | n.s. |
| foreign (cross-task donor, 0 same-task) | 144 | n.s. |
| permute_once (slot roll) | 141 | n.s. |
| reset_k=8 | 113 | p=6e-5 |
| write_every=4 | 65 | p=1e-15 |
| freeze_k=1 | 56 | p=8e-18 |
| reset_k=2 | 38 | p=4e-25 |

The pre-registered falsification fired: any *statistically mature* state fully
substitutes for accumulated memory; *immature* states collapse toward bypass.

## 3. Offline battery (T0.2; 5,659 paired segments, 4 suites)

- bypass +0.0081 MSE [CI +0.0075, +0.0088]; prior +0.036; **shuffled content −0.00002**.
- Burn-in curves **order-invariant** (forward ≡ reversed ≡ shuffled to 4 decimals):
  value is a function of write COUNT, not content — integrator signature.
- λ dose-response minimum exactly at trained λ=1; live and shuffled curves identical.
- Benefit is **front-loaded**: bypass−live gap +0.019 at decisions 0–4 shrinking to
  +0.004 at 32–64 — a clock is most useful early, a content memory would grow.

## 4. Blackout stress test (T0.5; goal+object pooled /200, blackout = decisions 8..8+D, black, both views)

| arm | D=0 | D=1 | D=2 | D=4 |
|---|---|---|---|---|
| live | 106 | 82 | 67 | 51 |
| live, writes suppressed during blackout | — | — | — | 55 |
| **foreign donor** | 99 | — | — | **62** |
| bypass | 24 | 23 | 13 | 18 |

The pathway bridges blackouts (51–62 vs 18) but **content does not**: the foreign
donor bridges at least as well as own memory, and suppressing writes during the
gap changes nothing. The content-null extends to the blackout regime.

## Interpretation

At 38% of co-training, memv1 functions as a **differentiable episode clock**: the
working state's norm/spectrum grows with write count and the co-adapted action
head heavily exploits that maturity signal (~45% of its fused input by magnitude);
episode-specific content contributes ≤5pp closed-loop / ≤2e-5 MSE offline anywhere
tested, including under observation blackouts. Four independent probes agree
(transplant grid, offline battery, order-invariance, blackout controls).

## Consequences for the plan

- Paper framing pivots to mechanism: "what a memory module actually learns under
  short credit horizons" — the clock finding + the diagnosis (4-decision BPTT,
  detached burn-in, Markovian suites) + redesign evidence.
- At 100K: rerun {live, prior, bypass, foreign, noisematch} (does content emerge as
  the gate grows?); T0.7 archaeology axis = live−foreign over training.
- T2.3 (credit-horizon branches) is promoted to the pivotal training-side experiment;
  T1.1 twin remains for the training-attribution row.
- T0.8 (occluded-cue benchmark) remains the decisive content-demand test —
  standard LIBERO cannot reward content even if it were stored.
