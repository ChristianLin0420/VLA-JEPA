# Adversarial Review: VLA-JEPA memv1 Experiment Plan

Verified against the codebase map plus direct spot-checks of the repo (`VLA_JEPA.py:430-432` guard, `recurrent_memory.py:61 _reset_parameters`, `GR00T_ActionHeader.py:262-264 sample_time`, `predict_action` at `VLA_JEPA.py:526-599`, surviving checkpoints, DROID episode-length distribution, LIBERO init files).

## Feasibility

1. **FATAL — T0.2d / Fig. 10 "temporal reach to ~640 decisions on DROID" is impossible.** Verified from `droid_lerobot/meta/episodes.jsonl`: exactly **1** episode of 92,233 reaches 640 decisions (4,480 frames); ≥300 decisions = 232 episodes, ≥150 = 1,787 — *before* excluding training-consumed episodes. The claimed "~500 genuinely unseen long DROID episodes" cannot support the panel as drawn. Fix: cap the depth axis at ~100–150 decisions (5,365 episodes ≥100) and re-title the figure.

2. **MAJOR — T1.3's "gated arm's own 20K checkpoint" no longer exists.** `keep_last_checkpoints: 3` already pruned it; surviving cotrain checkpoints are only {27752, 30000, 34729} (verified by `ls`). Same defect hits T2.6's "seed-42 counterparts already exist (live run's first 20K)" for any success-rate readout (TB loss curves survive, checkpoints don't). Fix: match FIFO/seed comparisons at step 27752, or run FIFO pilots to ~28K.

3. **MAJOR — T0.7 "Emergence of memory use" has an unrecoverable hole exactly where emergence happened.** Cotrain saves at 6988/10000/13911/20000/20847 are pruned; the gate's growth from 0.0125→~0.022 occurred largely in the destroyed 5K→27.7K window. T0.0's archiving fixes the future, not the past. Fix: reframe the figure as sparse anchor points (stage1-5K, 27.7K, 30K, 34.7K, then 10K-dense) and say so.

4. **MAJOR — T0.1 counterfactual cost is off by ~7×.** `predict_action` (`VLA_JEPA.py:544-549`) runs the full Qwen3-VL-2B encode on every call; a second per-decision call doubles the dominant compute — roughly +100% server time, not "~10–15%". Making it cheap requires a token-reuse refactor of `predict_action`, which exceeds H5's ~90-LOC server-only claim. Fix: either budget ~2× serve time for the counterfactual arm or add an explicit `cached_qwen_tokens` kwarg to H1 and count the LOC.

5. **MINOR — T0.2's "sample beyond the offset" unseen-DROID enumeration is glossed.** Segment index → episode is mediated by per-(epoch,index,seed) RNG and uniform 1/7 dataset draw (`datasets.py:1668-1683, 1694-1696`); enumerating consumed episodes requires replaying the sampler RNG for all `completed_steps×8` indices, and at the 100K headline checkpoint ~114K DROID draws leave only ~30% of episodes untouched — fewer among the long ones the analysis needs. Fix: write the sampler-replay enumerator into H7 explicitly and verify unseen-long-episode counts before promising N=500.

6. **MINOR — H7's `(t, noise)` pass-through at `GR00T_ActionHeader.py:262-264` contradicts "offline scripts (new, no trainer changes)".** It edits a core model file (verified: `sample_time` lives there). Fix: move it to H1/H2's changed-file ledger; it's fine, just count it.

7. **MINOR — T2.1 relies on unverified simulator surface.** `obj_of_interest` exists in LIBERO (`bddl_utils.py:94-128`, verified), but the plan's qpos-edit-plus-`sim.forward()` mid-episode with `env.check_success()` guard is untested infrastructure; also confirm the edit doesn't trip `done` spuriously. Fix: 1-task smoke before committing the grid.

8. **MINOR — T1.2/T2.3/T2.4 "branch from exported step_40000" conflates weight export with resumable state.** T0.0 archives `model.safetensors` only; branch runs are therefore warm-starts with fresh optimizer/scheduler, not resumes. Paired reference branch makes this internally consistent, but say it — and archive `optimizer.bin` for step_40000 if a true resume is ever wanted.

## Validity

9. **MAJOR — §2's "pairing is free and exact" is contradicted by the plan's own noise citation.** If episodes were bitwise deterministic, the two identical allv2 repeats (`SLURM/vlajepa_evalall_601583{4,6563}.out`) would agree exactly; they differ by 6 pp on libero_10. So closed-loop rollouts are not reproducible (GPU nondeterminism + chaotic divergence), cross-mode "episode k identical" holds only at t=0, and validation-contract items (3) and (4) ("λ=0 reproduces bypass episode-for-episode") will likely fail as written even with correct code. Fix: claim pairing on (task, init state, noise seed) only, demote bit-checks to single-forward fixed-observation tests, and let T0.0's ICC decide how much the pairing actually buys.

10. **MAJOR — sample size is pre-registered before the power analysis that justifies it.** The primary endpoint locks 50 trials/task, but the MDE comes from T0.0's k=3 repeats — an SD estimate with 2 degrees of freedom (true SD could be 0.5–6× the estimate). If MDE > plausible effect (gate 0.022 ⇒ ~2% injection), the pre-registered primary is dead on arrival. Fix: pre-register the *decision rule* (endpoint + test), set N after calibration, k≥5 repeats.

11. **MAJOR — T0.4 and T0.3 are underpowered at 10 trials/task.** All transplant/reset arm differences are bounded above by live−bypass (itself possibly a few pp); at 200 episodes/arm with ±6 pp/100-episode suite noise, the isotonic ordering live > same-task > cross-task ≥ noise and the Cochran-Armitage trends cannot resolve. Fix: run T0.4/T0.3 only on libero_10 at 30–50 trials for the 3–4 decisive arms; cut the rest.

12. **MAJOR — headline live−bypass conflates memory content with head co-adaptation to *any* injection.** The head trained with fusion always on; bypass is off-distribution for it. The plan's own decomposition (prior−bypass vs live−prior) contains the answer, but Table 1's headline and the abstract-level claim ("memory path causally contributes") ride on live−bypass. Fix: make live−**prior** the content claim and live−bypass the pathway claim, explicitly, in the pre-registration.

13. **MINOR — T0.5's placement falsification is confounded.** At d0=0 live still injects the trained prior (prior−bypass ≠ 0), so the live−bypass gap need not vanish when memory is "empty"; the control conflates empty-memory with no-injection. Fix: run the d0=0 arm as live vs *prior*, not live vs bypass.

14. **MINOR — T0.0 hypothesis (a) mislabels the metric.** "injection ratio grows 0.0125→0.0219 from safetensors" — those numbers are tanh(gate); injection ratio ‖tanh(g)·residual‖/‖consumer‖ needs activations and cannot be backfilled CPU-only from safetensors. Fix: backfill gate only; measure injection ratio via the T0.2 replay per checkpoint.

15. **MINOR — T0.6 knock-out "in-distribution" claim is wrong.** `recurrent_memory.py:221-223` zeros *whole rows* (all 8 slots) for inactive batch entries; a state with one zeroed slot among seven live ones never occurs. Harmless, but drop the "exactly what inactive rows produce" justification.

16. **MINOR — T0.1 demand stratification is n=10.** Spearman over 10 libero_10 tasks with permutation p cannot support "demand predicts gain" as a figure-level claim. Fix: pool per-task Δ across all 4 suites (40 tasks) or demote to appendix.

17. **MINOR — quoted JEPA cosine trajectory (0.750→0.767) disagrees with the logged values (0.755@50 → 0.796@34700).** Trivial but it's in the claim-scope paragraph reviewers will read first. Fix the numbers.

## Redundancy / bloat

18. **MINOR — two replay harnesses do one job.** `replay_modes.py` (T0.2, teacher-forced writes on cached tokens) and `replay_memory_states.py` (T0.6, re-runs `predict_action` per decision) produce the same state trajectories; T0.6's states, diagnostics, and knock-outs can come from the T0.2 replay with H3 enabled. Fix: one replay engine, two analysis frontends.

19. **MINOR — T0.3's reset-K=1 arm is action-identical to prior mode by construction** (write lands after the action, state discarded before the next decision), so it "verifies" nothing independent. Fix: drop K=1, keep K∈{2,4,8,16}.

20. **MINOR — the implementation-surface accounting undersells itself.** H7 says "3 self-contained scripts" but lists 5+ plus an `analysis/` tree; H5's 90 LOC must cover 11 modes, donor-bank IO, state dumps, counterfactual dispatch, and response extras — 200+ LOC realistically. Not fatal, but the "~200 LOC total" headline will be quoted back at you. Fix: restate as ~350–400 LOC or trim modes (drop `noisematch` or `foreign_sametask` from Tier 0).

## Gaps

21. **MAJOR — strongest open reviewer objection: no task where memory is *required by design*.** The plan's own Evaluation-Plan.md:127-138 says standard LIBERO is insufficient and Level-C claims need a versioned memory-dependent benchmark; the plan substitutes an inference-time blackout (T0.5) that the model was never trained under — a reviewer will call it a synthetic OOD-robustness probe, not a memory benchmark, and note the training pipeline (4-decision credit, detached burn-in, gate 0.022) was never given a reason to store anything. Fix: add one cheap in-simulator delayed-cue/occluded-goal LIBERO variant (init-state edit + hidden object after decision k) as the versioned memory task, even at pilot scale.

22. **MAJOR — the entire attribution story rests on n=1 training run per arm.** T1.1 twin vs memv1 is one seed each; seed variance of full runs (T2.6) is optional Tier 2, and the plan's own docs warn the 78/92/98/96 reference is "a one-seed reference, not a confidence interval". Fix: promote a 20K-step 2×2 seed cell (T2.6, descoped) into Tier 1 as a mandatory noise floor for the twin comparison.