# Memory-Benchmark Adoption Proposal (verified 2026-07-04)

All load-bearing availability claims below were re-verified today by fetching the repos/dataset pages. Notable verification deltas vs. the candidate briefs: **LIBERO-Occ's repo is skeletal** (1 star, 2 commits, placeholder "TODO" install URL, no occlusion-pipeline code published) — downgraded. **RoboMemArena is healthier than briefed** (309 stars, active June 2026, confirmed LIBERO fork + (T,7) EE actions + OpenPI runtime) but has **no license stated** — watchlist. **RoboMME is confirmed ICML 2026 Spotlight with a live CVPR 2026 challenge** — recognition upgraded. MIKASA's LeRobot-v3 dataset (`mikasa-robo/mikasa-robo-vla-lerobot`, 1.1k downloads/mo) and LIBERO-Mem's 211 GB HF dataset both confirmed live.

---

## 1. Ranked shortlist

Scores 1–5; integration cost is 5 = cheapest for our MuJoCo/robosuite/EGL + Franka + 2-RGB + language websocket harness.

| Rank | Benchmark | Content-memory demand | Integration cost | Demos for FT | Recognition | Verdict |
|---|---|---|---|---|---|---|
| 1 | **LIBERO-Mem** (arXiv 2511.11478, AAAI 2026) | 5 — visual aliasing by construction (counts, identity swaps, occlusion) | **5 — same simulator, robot, BDDL, HDF5, EGL** | 3 — 1,200 human demos (100 train/task), 211 GB verified on HF | 3 — AAAI 2026, young repo (26 stars, 2 commits) | **Adopt first** |
| 2 | **MIKASA-Robo-VLA** (arXiv 2502.10550, v1.0.0 May 2026) | 5 — purest cue-shown-then-removed design; chance-level for memoryless policies; per-mechanism isolation | 2.5 — ManiSkill3/SAPIEN Vulkan headless unproven on our cluster | **5 — 22,500 oracle trajs in native LeRobot v3, verified; regenerable oracles** | 4 — ICLR-track paper, 114 stars, muVLA/MemoryVLA already report on it | **Adopt second (gated on Vulkan smoke test)** |
| 3 | **RoboMME** (arXiv 2603.04639) | 5 — "identical observations, different histories, different actions" is the design axiom; 4-way taxonomy | 2.5 — same SAPIEN/Vulkan risk (Docker w/ CUDA 12.8 provided); Franka + front/wrist 256px matches our obs spec | 3 — 1,600 scripted HDF5 demos verified; waypoint scripts allow more | **5 — ICML 2026 Spotlight + CVPR 2026 challenge; MemoryVLA/SAM2Act+/MemER baselines in-repo** | **Adopt for eval protocol; shares infra with #2** |
| 4 | **RoboCerebra** (arXiv 2506.06677) | 3 — Memory-Exploration/Execution splits are genuine but diluted (planner-centric protocol, many Markovian subtasks) | 4 — LIBERO/MuJoCo, verified; ~3k-step episodes stress harness wall-clock | 4 — 1,000 human teleop demos on HF | 3 — NeurIPS-track, 68 stars | Optional long-horizon stressor |

**Losers (one line each):**
- **RoboMemArena** — verified real and LIBERO-compatible (309 stars), but no stated license (blocks dataset use), 1,000+-step episodes, zero external adoption; **watchlist, revisit in a quarter**.
- **RMBench** — dual-arm Aloha embodiment breaks our 7-DoF single-Franka action head; mine its PutBackBlock/BatteryTry designs instead.
- **MemoryBench (SAM2Act+)** — CoppeliaSim has no EGL path (Xvfb/VirtualGL pain), 3 tasks, saturated at 94.3%; reference designs only.
- **LIBERO-Occ** — verified repo is unpublishable-against (2 commits, no pipeline code, TODO install URL), and static from-frame-0 occluders don't force memory reads.
- **KC-VLA** — 4 clean probes but 100 eps/task, single-author repo, SAPIEN risk; use as task-design reference.
- **MEMBOT** — no code or benchmark released; **steal the observation-dropout protocol** (see portfolio).
- **MemoryVLA / muVLA** — methods, not benchmarks; adopt muVLA's two-benchmark protocol and replicate MemoryVLA's Guess-Where/Seq-Push-Buttons as BDDL if we build.
- **POPGym / Memory Gym / Memory Maze** — CPU unit tests for the memory cell, not manipulation benchmarks; MIKASA already ported them to tabletop.
- **RoboCasa / BEHAVIOR / Habitat / native ManiSkill3 / "MemVLA" etc.** — verified negative leads; no adoptable memory tasks.

---

## 2. Adoption plans

### 2.1 LIBERO-Mem (rank 1)

- **Install (1–2 days):** Clone `github.com/libero-mem/libero-mem`; diff their vendored `thirdparty/` robosuite/robomimic against our pinned versions; copy the 10 BDDL files (`libero/libero/bddl_files/libero_mem/`) + scene assets into our LIBERO install; EGL path unchanged. Pull HF dataset (`libero-mem/LIBERO-Mem`; use `-Raw` if we don't need masks/depth — full set is 211 GB).
- **Adapter work (2–3 days):** Register `libero_mem` suite in the websocket eval server; port their sequential subgoal checker (with overshoot detection) as an additional metric alongside binary success; reuse existing LIBERO HDF5→LeRobot converter for demos.
- **Eval protocol:** 10 tasks × 20 seeds, report success + subgoal-completion ratio. Run our **live / foreign / bypass discriminator** per checkpoint: (a) live recurrent memory, (b) memory state transplanted from a foreign episode, (c) zeroed/frozen memory. Prediction that validates the benchmark for us: live ≫ foreign ≈ bypass here, vs. live ≈ foreign ≈ bypass on vanilla LIBERO (the episode-clock signature). Episodes up to 700 frames directly stress the 8×512 horizon.
- **Fine-tuning path:** 1,000 train demos → LeRobot; co-train with our existing LIBERO mixture (~1:4) under the TBPTT recurrent pipeline we just landed; hold out the 20 val demos/task.
- **GPU-h estimate:** eval ~5–10 GPU-h per checkpoint × 3 discriminator conditions; fine-tune ~200–400 GPU-h per run on 8×H100; ~600–900 GPU-h total for a 2–3 variant sweep.
- **Calendar:** first zero-shot + discriminator numbers in ~1 week; fine-tuned results in 2–3 weeks.
- **Risks:** keyboard-teleop demos are jerky (inspect action smoothness before BC); 2-commit repo means budget a day for subgoal-eval rough edges.

### 2.2 MIKASA-Robo-VLA (rank 2)

- **Install (1-day gated smoke test + 1–2 days):** Mirror wheels, `uv sync --frozen`. **Gate:** Vulkan headless smoke test on an H100 node — bind `nvidia_icd.json`/`10_nvidia.json` ICDs into the container (ManiSkill3 headless docs + the Maniskill3-Singularity recipe). If this fails after 2 days, stop and fall back to rank 1 + portfolio wrapper.
- **Adapter work (3–5 days):** Gymnasium→websocket bridge (~200 lines); map `pd_ee_delta_pose` ↔ our 7-DoF delta-EE convention; configure a wrist+front 2-view camera layout per task (verify availability); pass through `LANGUAGE_INSTRUCTION`.
- **Eval protocol:** select a ~12-task battery spanning the 10 memory types and Short/Medium/Long splits (anchor tasks: ShellGame, RememberColor, TakeItBack, ChainOfColors — known chance-level for memoryless policies). Adopt muVLA's protocol: train-task success + **held-out memory-task generalization** + LIBERO as Markovian control; run live/foreign/bypass on the anchors.
- **Fine-tuning path:** `mikasa-robo-vla-lerobot` (LeRobot v3) loads directly into our pipeline — zero conversion. Domain gap vs. LIBERO means fine-tune/co-train, not zero-shot eval; oracle PPO motion style differs from human demos (note in analysis).
- **GPU-h estimate:** ~500–1,000 GPU-h for fine-tuning on a 22.5k-traj subset; eval is cheap (GPU-parallel sim), ~5 GPU-h/checkpoint.
- **Calendar:** smoke test day 1; trained + evaluated in 2–3 weeks from a pass.

### 2.3 RoboMME (rank 3 — piggybacks on 2.2's infra)

- **Install:** their Docker (CUDA 12.8) → enroot/Singularity conversion; SAPIEN/Vulkan test shared with MIKASA — evaluate both together.
- **Adapter work (2–4 days on top of 2.2):** env→websocket bridge variant; front+wrist 256×256 matches our 2-view spec almost exactly; HDF5→LeRobot converter (~1 day, format documented in `doc/h5_data_format.md`).
- **Eval protocol:** their fixed 100/50/50 seed splits, 50 test episodes/task. The **512-token memory budget protocol matches our 8×512 module exactly** — report our design as an additional MME-VLA variant against their TTT/RMT recurrent baselines, pi-0.5, MemoryVLA, SAM2Act+, MemER. Live/foreign/bypass is most diagnostic on the Reference suite (cue shown once, never repeated).
- **Fine-tuning path:** joint multi-task on 1,600 demos (their official protocol); regenerate more via waypoint scripts if thin.
- **GPU-h estimate:** ~300–500 for multi-task fine-tune; eval 16 tasks × 50 eps × up to 1,300 steps ≈ 20–40 GPU-h per checkpoint per condition.
- **Calendar:** 2–3 weeks, overlapping with 2.2. The CVPR 2026 challenge is a visibility opportunity if our numbers are competitive.

### 2.4 RoboCerebra (rank 4 — optional)

- Same MuJoCo/LIBERO stack; main work is extending websocket rollout length to ~3k steps and defining a flat-policy protocol (official baseline is a hierarchical VLM planner — document the deviation). 1,000 human demos via existing HDF5→LeRobot path. ~1 week integration, ~200 GPU-h. Adopt only if we want a retention-duration result (does the 8×512 state survive 3k steps) after ranks 1–3 land.

---

## 3. Recommended portfolio: adopt 2, build 1 instrument, retire 1 plan

**Adopt:**
1. **LIBERO-Mem** — primary external evaluation + first content-demand fine-tune. Credible (AAAI 2026), zero simulator risk, and directly tests the episode-clock hypothesis on checkpoints we already have.
2. **MIKASA-Robo-VLA + RoboMME as a pair** (one Vulkan smoke test unlocks both): MIKASA supplies the *training corpus* (22.5k LeRobot-v3 trajectories — the only candidate with enough data to co-train a 2B model under content demand); RoboMME supplies the *reviewer-facing eval frame* (ICML Spotlight, published baselines, and a 512-token budget protocol that makes our 8×512 directly comparable). If forced to pick one: MIKASA for a training paper, RoboMME for an analysis/audit paper.

**Build (small):** a **partial-observability wrapper kit** on our existing LIBERO harness — `geom_rgba` alpha cue-hiding, per-camera blackout/freeze windows, mocap-driven occluders with `contype/conaffinity=0` (MEMBOT's dropout protocol done properly). ~2–4 days. This is the only instrument that isolates the memory variable while holding tasks, visuals, and demos fixed (existing LeRobot data stays valid — occlusion is an observation-side transform), so it is our controlled-ablation companion to every external benchmark. Not publishable alone; publishable as the audit methodology.

**Retire/fold: in-house libero_mem_v0 (T0.8 occluded-cue).** Honest comparison: LIBERO-Mem now covers ~70% of its purpose (occlusion, counting, identity aliasing, in our exact stack) with external credibility we cannot manufacture, and it costs us weeks of task engineering plus demo collection to reach parity — reviewers will ask "why not LIBERO-Mem?" and we would have no good answer. What libero_mem_v0 uniquely offered — scripted demo generation at scale and precisely scheduled cue-visibility windows — survives as features of the wrapper kit above, not as a standalone benchmark. Revive the full design only if LIBERO-Mem's tooling proves broken in practice.

---

## 4. Best first move: LIBERO-Mem, started this week, with the Vulkan smoke test running in parallel

Reasons: (1) it is the only genuinely content-demanding benchmark that is **drop-in for our proven stack** — same MuJoCo/robosuite/EGL, Franka, BDDL, dual-RGB+language, 7-DoF delta-EE, HDF5 — so it produces the decision-critical measurement (does live memory beat foreign/bypassed memory when content is causally required?) in about a week with near-zero infrastructure risk; (2) that measurement determines everything downstream — if our trained memory reads content under demand, the story is fine-tuning/co-training (→ prioritize MIKASA's data); if it stays a clock even here, the story is architectural/training-signal (→ prioritize RoboMME's representation-comparison frame) — so it is the cheapest experiment that disambiguates the two research branches; (3) its verified dataset and AAAI 2026 provenance make whatever we find immediately citable. The 1-day SAPIEN/Vulkan smoke test costs almost nothing to run concurrently and de-risks ranks 2–3 before we need to commit.

Concrete week 1: days 1–2 install + BDDL/version audit + Vulkan smoke test in parallel; day 3 zero-shot eval of current checkpoints; days 4–5 live/foreign/bypass discriminator sweep + subgoal-metric port; week 2 demo conversion and first co-training run (~200–400 GPU-h).