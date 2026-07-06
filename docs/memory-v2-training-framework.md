All grounding is read — both report chapters, the codebase map, `VLA_JEPA.py`, `fusion.py`, `recurrent_memory.py`, and both configs — and the four lens reports are synthesized below into the single memv2 proposal, with cross-lens conflicts resolved.

---

# memv2 — Masked-Reconstruction Memory Training: THE Design

**Status:** design only. Grounded in `docs/memory-experiments-report.md` (ch. 1–2), `docs/memory-codebase-map.md`, `starVLA/model/framework/VLA_JEPA.py`, `starVLA/model/modules/memory/{recurrent_memory,fusion,state}.py`, `scripts/config/vlajepa_memv1_{video,stage1,cotrain}.yaml` (all under `/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA/outputs/wt-memexp`).

**Design axioms (from the closed causal chain):** storage works (probes 0.68), credit works (T2.3), moderate demand alone fails (T-FT), the read collapses to a maturity pacemaker by stage-1 ~5K under BC. Therefore every element below is scored on one question — *does it force gradient through the policy's own read into episode-specific slot content, before the shortcut can form?* — and every loss target is a frozen-`vj_encoder` latent (stationary teacher, no EMA, no pixels, no retrieval DB).

---

## 1. Executive summary

memv2 turns observation masking into the memory's own objective and rebuilds the read so the clock shortcut is structurally worthless. During training, whole decisions are blacked out (the exact corruption operator the eval blackout client already uses): on a blind decision the behavior-cloning loss still demands the expert's episode-specific action, and a new masked-reconstruction loss demands the true scene's frozen-V-JEPA2 latents — decoded by the *existing* JEPA predictor conditioned on `policy_tokens`, the very tensor the DiT consumes — so both losses are unsatisfiable unless content transits `policy_memory_fusion`. Simultaneously the read is re-architected (`SparseKeyMemoryFusion`): write-time content keys make retrieval content-addressed, the residual is whitened so it physically cannot carry maturity statistics, and the pacemaker signal the head genuinely needs (−44 pp when removed) is handed over for free as one explicit, ablatable time token — *shortcut drainage*, not shortcut fighting. Memory training begins in the video stage (multi-window SSV2 + DROID-as-video unrolls, no actions needed), so the read is born content-routing before BC ever touches it; a residual-anchored InfoNCE with same-task negatives guards against task-mean collapse; and the live-vs-foreign/shuffled discriminator — our proven sub-behavioral instrument — is promoted to an always-on training meter with pre-registered go/no-go gates, including a stage-1 abort at 5K that caps downside at under a day of compute.

---

## 2. The objective suite

Four losses total; two are new. Everything else proposed by the lenses is cut (see §2.5).

| # | Loss | Operator | Steps active | Weight | What it certifies |
|---|---|---|---|---|---|
| L_act | blind + sighted BC | existing flow-matching MSE, unchanged | all supervised decisions, incl. masked | 1.0 | the DiT itself demands content |
| L_rec | masked latent reconstruction *through the policy read* | existing `F.l1_loss` vs frozen teacher | masked decisions (replaces wm forward there) | 0.5 (stage 1) → 0.2 (co-train); grad-scale α=0.1 into `policy_tokens` | content reaches the DiT's input tensor |
| L_nce | residual-anchored episodic InfoNCE | new, small heads | robot supervised decisions | 0.2 | content is *episode*-specific, not task-mean |
| L_wm | JEPA world loss | existing, unchanged | unmasked robot steps ×0.1; video ×1.0 | as today | predictor stays warm (unchanged role) |

### 2.1 L_act — blind-decision behavior cloning
**Formulation.** The dataloader (`sample_segment`, `datasets.py:1717–1795`) emits a per-timestep `mask_plan` (deterministic per (epoch, index, seed), same discipline as commit 25e1882). On a masked supervised decision, `_forward_one` blacks both views of `example["image"]` (Qwen input) *and* both views of the VJ2 clip before `vj_processor` — the identical operator to the serve-time blackout protocol (§3.3 of the report) — while the sample carries `video_clean` for targets. The flow-matching action loss (`GR00T_ActionHeader.py:316`) is computed unchanged on the true expert chunk [B,7,7].
**What it trains.** On a blind step the embodied tokens carry only instruction + black-frame content, so the only route from "where is the object in *this* episode" to the DiT is the fusion residual. This is the demand tie: the DiT's own cross-attention receives gradient toward the content directions in `policy_tokens`.
**Anti-leakage/anti-collapse.** Vision leak closed by construction (frames black). Residual leak = task-prior: instruction + phase → task-conditional mean action, which is exactly the memv1 clock solution and is *good enough* on Markovian steps (foreign donors bridged blind gaps equally well, 51 vs 62). L_act is therefore decisive only where blind actions are episode-specific — which is why it never ships alone; L_rec and L_nce manufacture episode-specific demand on every blind step regardless of data Markovianity. Ground-truth targets → no collapse mode.
**Known vs novel.** Known (observation dropout in BC; the dead `memory_direct_context_dropout` knob anticipated it). The placement inside burn-in-8/seg-4 recurrent segments with the foreign-gap meter attached is ours.

### 2.2 L_rec — masked reconstruction through the policy's conditioning tensor (the seed idea, made read-coupled)
**Formulation.** On each masked decision t:
- Targets: `gt_states = sg(vj_encoder(video_clean_t))` → [B,768,2048], exactly the existing `_compute_world_loss` teacher path (`VLA_JEPA.py:301–307`), frozen, no-grad.
- Decoder: the **existing `vj_predictor`** (12-block `VisionTransformerPredictorAC`). Its `input_states` are replaced by a learned `wm_mask_token` (`nn.Parameter([2048])` broadcast to [B,768,2048]; RoPE positions unchanged, so the predictor knows *where/when* it predicts — V-JEPA/MAE-style latent masking).
- Conditioning: **`mem_cond = mem_cond_adapter(policy_tokens)`** — a learned token mixer 32→24 plus a zero-init low-rank channel map (2048→256→2048), producing predictor-compatible conditioning tokens. `policy_tokens` is the post-fusion output of `policy_memory_fusion` (`VLA_JEPA.py:392–394`) — the exact tensor the DiT cross-attends. On unmasked steps the wm path is untouched (`qwen.action_tokens` conditioning, as today).
- Loss: `L_rec = F.l1_loss(predicted_states, gt_states)` (`VLA_JEPA.py:309`, unchanged operator), logged separately from `wm_loss`; the recon gradient entering `policy_tokens` is scaled by α≈0.1 (a scale-gradient op) so recon cannot destabilize BC early.

**What it trains.** Gradient enters through `policy_tokens`, i.e. through the fusion's `output_projection ∘ attention ∘ value_projection`, the content gate, and — within BPTT-4 — recent writes. Because blind embodied tokens are near-constant, the loss specifically pressures the memory-side projections and slot values to expose content: the exact parameters the pacemaker solution leaves unused. On blind steps a step counter carries ~0 bits about 768×2048 episode-specific latents — the shortcut is information-theoretically dead *for this loss*.
**Anti-leakage, three channels closed:** (1) current observation — frames black, targets from `video_clean` only; masked runs of D≥2 at stride 7 mean no 1-frame window overlap can answer the reconstruction; (2) instruction/task prior — the residual threat; countered by L_nce and *measured* continuously by the foreign gap (if `L_rec(foreign) ≈ L_rec(live)`, the decoder regresses the task template); (3) decoder memorization — the decoder is the shared, already-busy predictor (also serving L_wm), not a dedicated lookup table, and the always-on bypass control (`L_rec` with `fusion_bypass=True`, a kwarg that already exists at `_forward_one:347` / `fusion.py:96`) must stay at the unconditional floor.
**Anti-collapse.** Frozen-teacher targets (constant-target collapse impossible); the collapse mode that remains is conditional-mean regression, which is L_nce's job and the foreign-gap's alarm.
**Known vs novel.** Masked latent prediction = V-JEPA/MAE; reconstruction-trained memory = MERLIN (pixel/VAE, RL, decoder sees the present). Novel and load-bearing: frozen-JEPA latent targets, present-excluded by whole-decision masking across the recurrence boundary, **decoded through the policy's own fused conditioning tensor** — converting "memory stores it" into "the policy's input provably carries it," the precise gap the discriminator exposed.

### 2.3 L_nce — residual-anchored episodic InfoNCE (the anti-collapse guarantor)
**Formulation.** Anchor `r_t = h(pool₃₂(residual_t))`, where `residual_t` is captured **pre-gate** inside the fusion (post-`output_projection`, pre-whitening/gating), pooled over the 32 consumer positions, projected to d=256, L2-normalized. Positive `z⁺ = g(mean-pool(gt_states_{t−Δ}))`, Δ≥4 decisions, from this episode's past (detached, frozen-teacher — one extra no-grad pooled cache per segment). Negatives: cross-GPU `all_gather` of detached targets + a small FIFO queue (N≈256; MoCo-style buffer of *training targets*, not a serving-time database), with **same-task different-episode hard negatives** enforced by a sampler constraint (enabled by the already-present `emit_episode_metadata: true`). `L_nce = −log softmax(r·z⁺/τ)`, τ=0.07. Robot stages only.
**What it trains.** The read residual must be episode-identifiable — the discriminator's contrapositive as a differentiable loss. Mean-regression cannot beat same-task negatives; this is what keeps L_rec honest on near-identical LIBERO scenes.
**Failure mode, named.** With per-device batch = 1 segment, in-batch negatives are scarce — the gather + queue is required engineering; if same-task hard negatives are rare, the loss degrades to task-ID discrimination (zero new pressure — memory already encodes task at 0.68). Monitor NCE accuracy split same-task vs cross-task; only the same-task number counts.
**Known vs novel.** CPC/InfoNCE known; anchoring on the *fusion residual pre-gate* (read-directed, not state-directed) is the novel part.

### 2.4 L_wm — unchanged
Kept exactly as-is on unmasked robot steps (×0.1 hardcode at `VLA_JEPA.py:403` becomes a named knob) and the video pass (×1.0). It exists in this table only to state that memv2 does not perturb it; on masked steps it is disabled (its inputs would be black frames) and L_rec replaces its forward — so the marginal FLOPs of L_rec in co-train are ≈ 0.

### 2.5 Cut, with reasons
- **Foreign-swap margin loss (L3):** on Markovian steps its only descent direction is *sabotaging the foreign branch* — a reward-hacked sensitivity that would poison our own meter. The foreign gap ships as the primary **metric**, never a loss. (All four lenses converged here.)
- **Separate world-model fusion (the Phase-3 `world_model_conditioning` as configured):** a second consumer with its own gate recreates the two-consumer trap one level up. The dormant flag's *revival point* (`VLA_JEPA.py:140–141`) is reused, but conditioning is `policy_tokens`, not a second fusion.
- **State-level NCE, past-action recall:** train write/retention — already-solved territory (0.68).
- **Lag-k recall head with lag embeddings:** masked-run *bursts* (D up to 4) create multi-decision retention demand with zero new modules; explicit lag-k recall is the pre-registered escalation if the depth-profile diagnostic shows a 1-step-buffer solution.
- **Patch/tube masking:** grey boxes are OOD to the VLM; whole-decision blackout matches the deployed instrument. Single-view masking kept as a config knob, off by default.

---

## 3. Architecture changes

### 3.1 Forward pass with masking (robot supervised decision t; ★ = new)

```
mask_plan[t] ──────────────┐ (dataloader; deterministic)
                           ▼
image/video ──(★ black if masked)──> _encode_qwen_tokens ──> action_tokens [B,24,2048]
video_clean ──(targets only)                              └─> embodied_tokens [B,32,2048]

RecurrentMemory.read(state) ─> tokens [B,8,512] + ★keys [B,8,128]
                           │
embodied_tokens ───────────┴──> ★SparseKeyMemoryFusion:
    q = qk_proj(LN(embodied))                        [B,32,128]
    scores = q · key_norm(state.keys)ᵀ / √128        [B,32,8]
    attn  = top2-softmax(scores)  (learnable temp, floor)
    r     = out_proj(attn · value_proj(state.working))   [B,32,2048]
    r̂     = LayerNorm_no_affine(r)              ← WHITENED: no norm/maturity carrier
    g_c   = σ(gate_mlp([proj64(r̂); max score; top1−top2 margin; attn entropy]))
    tap   = time_mlp(sinusoid64(log(1+steps)))   [B,1,2048]  ← the clock, paid off
    policy_tokens = [embodied + tanh(γ_c)·(g_c ⊙ r̂) ; tap]   [B,33,2048]
                           │
        ┌──────────────────┼───────────────────────────┐
        ▼                  ▼ (masked steps only)        ▼ (unmasked steps only)
   DiT action head    ★mem_cond_adapter(policy_tokens) vj_predictor(input_states,
   → L_act [all]      → vj_predictor(★wm_mask_token,     action_tokens) → L_wm ×0.1
                         mem_cond) vs sg(vj_enc(video_clean))
                      → L_rec        (pre-gate r → pooled anchor → ★L_nce)

write: update_mask_t = update_mask & ~masked   (no black-frame writes, phase A)
       count_mask_t  = update_mask             (★ decision clock always ticks)
       working ← convex gated update (unchanged)
       ★keys  ← (1−ḡ)·prev_keys + ḡ·tanh(key_proj(context))   (bound to write source)
```

BPTT/detach logic (`forward_sequence`, `VLA_JEPA.py:432–528`) is unchanged: seg 4 / BPTT 4 / burn-in 8, `memory_detach_burn_in: true`. The mask-sampling constraint (first supervised decision always unmasked; burn-in never masked) guarantees a write→read pair inside every BPTT window.

### 3.2 Read-path redesign — `SparseKeyMemoryFusion` replaces `ResidualMemoryFusion`
Fixes the two diagnosed defects: (1) permutation-invariant read with keys derived from slot values (`fusion.py:104–111`; `slot_ids` appear only in the write query, `recurrent_memory.py:274–275`) → **write-time content keys** (`key_projection: Linear(512→128)` inside the FP32 write block; `MemoryState` gains `keys [B,8,128]` fp32, learned `initial_keys [8,128]`, `schema_version: 2`, export guard updated); a foreign state now presents *wrong keys*, not just wrong statistics, and slot permutation is no longer a read no-op. (2) The single scalar `tanh(gate)` (`fusion.py:42,113`) coupling time+content → **two-head split**: a whitened, per-token-gated content channel that physically cannot carry residual magnitude (the dominant clock carrier — injection ratio 0.45, norm saturating at 17.4), plus one explicit **time-tap token** from `log(1+steps)` (log features blunt the DROID extrapolation failure, §3.7). The tap is *shortcut drainage*: the −44 pp pacing benefit is real, so hand it over through a dedicated, exactly-ablatable channel and leave content as the only non-redundant thing slots can offer. No mask-bit input to the read (rejected: invites the complementary "never read when sighted" shortcut).

Sharp retrieval is top-2 softmax with learnable temperature (entmax15 as an alternative; top-k has zero new dependencies) — softmax over 8 similar slots ≈ mean slot ≈ a maturity statistic, which is what we're escaping.

### 3.3 Reused vs new

| Reused unchanged | Modified | New |
|---|---|---|
| Qwen3-VL (frozen), vj_encoder (frozen teacher), DiT head, flow-matching loss, `_compute_world_loss` targets, TBPTT/segment machinery, blackout operator, cache/replay + discriminator engine, websocket state plumbing | `RecurrentMemory` (+`key_projection`, `initial_keys`, `count_mask`), `MemoryState` (schema 2, `keys`), `vj_predictor` (accepts mask-token input states; no structural change), `sample_segment` (+`mask_plan`, +`video_clean`), `_forward_one`/`forward_sequence` (mask plumbing, loss switch), trainer (knobs + the `memory/*` diagnostics-clobber fix, `VLA_JEPA.py:427–430`), `predict_action`/serve modes | `SparseKeyMemoryFusion` (≈2.3M, replaces 3.42M), `wm_mask_token` (2K), `mem_cond_adapter` (≈1.1M, zero-init), L_nce heads h,g (≈1.0M), time-tap MLP (in fusion count) |

**Params:** memory subsystem ≈ 7.4M total (memv1: 6.32M) — under the existing <10M cap test. **Compute:** co-train VLA pass ≈ +0–10% (L_rec replaces the wm forward on masked steps; +1 no-grad teacher pass per segment for NCE targets); video pass grows with multi-window unrolls; net co-train ≈ +20–30% step time. Stage 1 gains a predictor forward it never had: +40–70% step time on a 10K-step stage (hours). Serve time +ε; state grows 4 KB/episode (keys).

**Serving:** `MEMORY_MODE` gains `tap_only` and `content_only` (each channel force-zeroed), mapping one-to-one onto the gate split so the H1–H7 harness can measure each channel causally; `keys` ride the existing session `MemoryState`.

---

## 4. Three-stage recipe

| | **Stage V — video (20K)** | **Stage 1 — robot (10K)** | **Co-train (100K)** |
|---|---|---|---|
| Data | 50% SSV2 multi-window (J=4×8 frames) + 50% DROID-as-video (J=8; the temporal-extent carrier, in-domain robot video, median 220 frames) | LIBERO robot segments, burn-in 8 + seg 4, stride 7 (as memv1 stage 1) | VLA pass: robot segments; video pass: the stage-V mixture (replaces single-window SSV2). No external memory-benchmark data by default (T-FT forgetting lesson); ≤10% LIBERO-Mem/MIKASA anchors only if gate V0 demands and the 30K vanilla eval holds |
| Masking | 1 of J windows blacked per sample | ramp 0→0.25 of supervised decisions over 2K steps, D=1 only, ≤1 per segment; burn-in never masked; first supervised decision always unmasked | hold 0.25–0.3 with bursts: D∈{1}→{1,2} (20K)→{1,2,4} (50K); writes at masked steps re-opened at 20K (phase B — teach `update_gate` to reject blank sources; watch p05/p95 spread) |
| Losses | L_wm ×1.0 (unmasked windows) + L_rec ×0.5 (masked window; fusion consumer = `action_tokens` — no embodied tokens exist on video; same shared weights) | L_act + L_rec ×0.5 + L_nce ×0.2 (L_wm off, as memv1 stage 1) | VLA: L_act + L_wm ×0.1 (unmasked) / L_rec ×0.2 (masked) + L_nce ×0.2; video: L_wm + L_rec. Two optimizer updates per step, unchanged — **no third pass** (a memory-only pass can't teach the policy read; it would re-run the memv1 lesion) |
| Trainable | memory_module, fusion, wm_mask_token, vj_predictor (3e-5); **Qwen frozen** (change vs memv1-video; warm-start from `vlajepa_memv1_video/final_model` keeps Qwen adapted *and* keeps discriminator Qwen-feature caches bit-comparable from here on) | memory_module, fusion, mem_cond_adapter, NCE heads, action_model @1e-4; predictor blocks 11–12 + mask token @3e-5; rest frozen | as memv1 co-train + new modules @1e-4; Qwen, vj_encoder frozen |
| Purpose | the read circuit is **born content-routing** before any BC gradient exists — the shortcut race (locked by stage-1 ~5K) is won by starting content learning where no clock reward exists | couple the content circuit to the policy consumers under BC, with the abort gate before the co-train spend | equilibrium: BC dominates, masking sustains demand, meters watch |
| Cost (8×H100) | ~10–14 h | ~6–7 h | ~72 h |

Every content objective is on from step 0 of every stage that trains the memory — curriculum order matters more than weights, because the competitor solution forms in ~5K steps. End-to-end ≈ 3.7–4.0 days vs memv1's ≈ 3.5 (+15%), with the stage-1 abort capping downside at <1 day.

---

## 5. Measurement plan

**Always-on training meters (near-free):** every logged step, evaluate (no backward) L_rec and blind-step L_act under `fusion_bypass=True` and under a maturity-matched **foreign** state (cross-rank detached state swap, equal `steps` — Qwen/encoder passes shared, only fusion+heads rerun). Log `Δbypass = L(bypass)−L(live)` (pathway meter) and `Δforeign = L(foreign)−L(live)` (content meter — the offline discriminator's logic promoted to a training curve). Plus: `memory/content_gate_mean`, `match_margin`, `tap_norm`, NCE same-task accuracy, update-gate p05/p95 — after fixing the co-train diagnostics clobber.

**Offline, per checkpoint (dependent job, cached-replay, single-GPU):** the chapter-2 discriminator (shuffled-content + foreign dMSE, masked decisions included in the cache), probe battery (task-id, phase, + new masked-content probe: decode the masked decision's object/goal from state), recon-vs-baselines (memory recon must beat prior-state recon and copy-last-visible by ≥20%).

**Go/no-go ladder (pre-registered thresholds; cheapest kill first):**

| Gate | When | Criterion | On failure |
|---|---|---|---|
| **G0 — demand audit** | before any training; CPU afternoon | regress expert action at t from (task, phase) vs (task, phase, init layout) on episode records; the gap = manufacturable demand ceiling | gap ≈ 0 on LIBERO ⇒ masking cannot work on this corpus; shift mixture toward LIBERO-Mem/MIKASA before training |
| **G1 — read unit test + leak audit** | stage-1 step 0; ~1 day | (a) state-replay read pretrain: freeze writes, train only fusion + recon path on cached memv1 state traces — the new read must extract stored content (foreign gap > 0); (b) with fusion bypassed, L_rec must sit at the unconditional floor (no leak) | (a) fails ⇒ read redesign is wrong, fix before spending; (b) fails ⇒ mask leak, fix dataloader |
| **G2 — stage-V exit** | 20K | video L_rec(live) beats both baselines ≥20%; probe on video-unrolled state above chance | drop stage V (critic V5), proceed with stage 1 warm-started from memv1-video final |
| **G3** | stage-1 2K | masked-segment shuffled-content dMSE leaves the memv1 null band (±3e-5) — i.e. *before* the memv1 shortcut-formation point | check L_nce hard-negative supply; extend 1K, else treat as G4 |
| **G4 — abort gate** | stage-1 5K | shuffled-content ΔMSE > +1e-3 (≥30× the memv1 null bound) AND live < foreign on masked-decision action-MSE | **split diagnosis:** recon-foreign gap > 0 but action-foreign gap = 0 ⇒ two-consumer trap despite ties (see §6, kill/escalate); both null ⇒ read still refuses under manufactured demand ⇒ ABORT co-train, <1 day spent |
| **G5** | co-train every 10K | discriminator gaps monotone-growing; pooled vanilla LIBERO within 5 pp of memv1 live at matched steps (37% @35K, 76% @100K); masked-decision action-MSE descending from prior-level (+0.036) toward ≤ +0.01 | forgetting breach ⇒ lower mask rate / burst prob; plateaued gaps ⇒ enable lag-k recall escalation |
| **G6 — endpoint** | 100K + blackout grid | **the pre-registered flip:** closed-loop blackout D≥2, live > foreign (McNemar, paired; memv1 baseline: 51 vs 62, foreign *better*); `tap_only` ≈ memv1-live on Markovian suites; offline foreign flips from p=0.46 to significantly worse than live; LIBERO-Mem/MIKASA: any nonzero (floors are 0/200) | live ≤ foreign with offline gap present ⇒ content read is real but not behaviorally decisive; claim retreats to certified offline read + diagnosis (still publishable) |

---

## 6. Risks and kill criteria

1. **The different-consumers trap (the central threat).** A reconstruction head can learn to read content while the action head stays a pacemaker — memv1's failure one level up; MERLIN never faced it because RL return eventually forced policy reads. memv2 ties the consumers four ways: (i) *supply tie* — L_rec decodes from `policy_tokens`, the DiT's own input tensor, through the same fusion instance, same content gate (no per-consumer gate scalar exists to defect through); (ii) *demand tie* — L_act on the same blind steps gives the DiT cross-attention gradient toward the content directions now present; (iii) L_nce anchors on the pre-gate fusion residual, the shared trunk; (iv) whitening + tap remove the rival solution the DiT would otherwise prefer. **Honest caveat:** a shared tensor does not force the DiT to *attend* — the two objectives could still partition `policy_tokens` into recon-channels and clock-channels. Detection is G4's split diagnosis; the pre-registered consumer-tie A/B (shared `policy_tokens` conditioning vs a private decoder arm) is run as a stage-1 side branch and is a publishable result either way. **Kill:** if the shared arm still splits, escalate once (recon decoded from DiT mid-features instead of `policy_tokens`); if that splits, the elegant-in-model program is falsified at this scale — stop.
2. **Task-mean collapse of L_rec.** Near-identical LIBERO scenes make a task-template latent score well; episode-specific variance may be a small L1 fraction. L_nce with same-task negatives is the structural counter; `Δforeign(L_rec)` is the alarm; per-token loss weighting by target variance is the reserve knob. **Kill trigger:** G4 both-null after the L_nce escalation.
3. **Leaks.** Named channels: masking predictor context but not Qwen images; targets computed from masked video; D=1 masks answerable from window overlap (stride 7 ⇒ enforce D≥2 for recon-bearing masks); deterministic mask schedules teaching schedule priors (randomized placement). G1(b) is the standing audit.
4. **Blind-rate forgetting.** T-FT precedent (304 → ~207 pooled under a naive mixture): mask rate capped ≤0.3, unmasked anchor per segment, vanilla LIBERO tracked at every G5. **Kill trigger for the schedule, not the design:** >5 pp sustained regression ⇒ halve mask rate.
5. **Clock re-entry through match statistics.** Early keys ≈ `initial_keys` ⇒ retrieval confidence correlates with step count; partially irreducible. Whitening blocks the magnitude route, the tap removes the incentive; the discriminator is the arbiter.
6. **Time-tap OOD** beyond trained horizons (steps ≤12 trained; 150+ on DROID): log-step features, and `tap_only`/`content_only` make the negative-transfer component removable and measurable — which memv1 could not do.
7. **Expert-demo ceiling (not fixable here).** Masking manufactures demand only for what the expert conditioned on; blind training is on-distribution but deployed closed-loop under compounding drift, and demos contain no information-gathering behavior. The discriminator can improve while blackout success does not (G6 retreat path). No RL/DAgger by project constraint — stated, not hidden.
8. **Stage-V transfer risk.** Fusion queries in stage V come from action tokens, not embodied tokens; write circuitry tuned on human video may mis-transfer. Mitigations: DROID-as-video carries the temporal load; shared weights + stage-1 re-anchoring; G2 makes stage V self-justifying and droppable.

---

## 7. Novelty statement (honest)

**Known, inherited openly:** masked latent prediction (V-JEPA/MAE — but their targets are recoverable from co-visible context; ours are severed from it by whole-decision masking across the recurrence boundary); reconstruction-trained memory (MERLIN — pixel/VAE, RL-driven demand, decoder sees the present); slot memory with content addressing and top-k reads (NTM/TTM — trained by task loss, the exact regime our chapter 1 shows collapsing, and never audited with a state transplant); memory banks for VLAs (MemoryVLA — success-rate deltas that our results show can be 100% maturity artifacts); observation dropout in BC; causal-confusion diagnoses (copycat — a sibling pathology; the pacemaker is a new instance, on internal state maturity).

**Claimed novel:** (1) *masking as demand manufacturing* for BC-trained memory — do(obs=∅) as a routing constraint where no return signal exists, not a pretext task; (2) *consumer tying* — masked reconstruction decoded through the policy's own fused conditioning tensor, designed against a measured two-consumer failure; (3) *shortcut drainage* — paying off a quantified pacemaker (−44 pp) through an explicit, ablatable time token plus a whitened content channel, rather than fighting it; (4) *the discriminator as emergence meter* — content-read emergence tracked as a training curve with pre-registered abort gates, below behavioral floors; (5) action-free pretraining of the identical write/read circuitry on video via the `{actions}`-marker prompt. Contributions (1), (3), (4) and the diagnosis survive even if the endpoint flip (G6) fails — the design is falsifiable at every gate, and the cheapest falsification costs a CPU afternoon.