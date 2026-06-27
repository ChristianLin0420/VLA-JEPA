# VLA-JEPA · Elegant Short- and Long-Term Memory
### A Design Proposal

> **Status:** proposal only — no code, configs, or training runs are changed by this document.
> **Goal:** give VLA-JEPA short- and long-term memory with the *smallest, cleanest* mechanism that works — **in-model, fully differentiable, fixed-size, reset-aware** — and explicitly **not** an agentic stack of external vector databases, write/eviction controllers, and paging logic.
> **Grounding:** every integration point cites a real `file:line` in this repo (re-verified). Every paper in §10 was web-verified (real arXiv id / venue).

---

## What changed from the previous draft

The earlier draft proposed a **three-tier agentic memory**: an external FAISS kNN bank, a hand-coded write controller, an eviction policy, a MemGPT-style paging controller, and periodic consolidation passes, plus a 14-column multi-timescale Parquet schema. That is a *second stateful system* bolted onto the model — heavy, brittle, non-differentiable at the retrieval step, and a poor fit for a real-time robot control loop.

This rewrite replaces it with **two tiers of in-model memory, each governed by a single closed-form update rule**:

| | **Agentic (rejected)** | **Elegant (this proposal)** |
|---|---|---|
| Where memory lives | external FAISS / vector DB | two small tensors on the model |
| Size | grows with episodes; needs eviction | **fixed** (chosen once) |
| Read | approximate kNN (cost grows with bank) | one attention / one matvec, **O(1)** |
| Write | controller + threshold heuristics | **one closed-form rule** per tier |
| Forgetting | eviction policy + consolidation job | a **learned decay gate** |
| Reset | purge/session management | **one tensor assignment** |
| Differentiable? | no (argmax/top-k ≈ zero gradient) | **yes, end-to-end** |
| Train/deploy | two systems, skew | **same code** at train and deploy |

Everything below follows from one principle: **memory should be a part of the network, trained by the losses already in the model — not a database the network queries.**

---

## Table of contents
1. [Executive summary](#1-executive-summary)
2. [Background: the current pipeline](#2-background-the-current-pipeline)
3. [The memory gap](#3-the-memory-gap)
4. [Design principles](#4-design-principles)
5. [The architecture — two tiers, two timescales](#5-the-architecture--two-tiers-two-timescales)
6. [Training](#6-training)
7. [Instrumentation (lightweight)](#7-instrumentation-lightweight)
8. [Evaluation](#8-evaluation)
9. [Roadmap, risks & honest tradeoffs](#9-roadmap-risks--honest-tradeoffs)
10. [References (verified)](#10-references-verified)

---

## 1. Executive summary

VLA-JEPA follows the **V-JEPA 2 → robot-cotrain** recipe [V-JEPA 2, arXiv:2506.09985; VLA-JEPA, arXiv:2602.10098]: a **frozen** V-JEPA2 encoder produces per-frame latents, an **action-conditioned latent predictor** forecasts the next-frame latent under **frame-causal attention** over a short clip (`T = num_frames // tubelet_size`, ≈8 frames), and a **flow-matching DiT** head emits continuous actions. The model is therefore **Markovian beyond the loaded clip**: no state persists across clips or episodes.

We add memory with **two in-model tiers**:

- **Short-term (working) memory** — a small set of **recurrent `[mem]` tokens** prepended to the predictor and carried across clips, with the backbone otherwise unchanged [RMT, arXiv:2207.06881; Block-Recurrent Transformers, arXiv:2203.07852]. This is a rolling, attention-readable summary of recent history.
- **Long-term memory** — a single **fixed-size associative fast-weight matrix** `W`, written by the **(gated) delta rule** and read by **linear attention** [Fast Weights, Schmidhuber 1992; Linear Transformers are Fast-Weight Programmers, arXiv:2102.11174; Gated DeltaNet, arXiv:2412.06464]. One outer-product write and one matvec read per step — slowly-decaying episodic/task context at **O(1) per step**.

Both tiers are **fixed-size, fully differentiable, and reset at episode boundaries by a single assignment**. They are supervised **for free** by the model's existing world-model loss and action loss — no new loss machinery, no external store, no controller. The **encoder and Qwen LLM stay frozen** throughout.

We also give a **design dial** (§5.6): the long-term tier can be made even more minimal (a second, slow-clocked `[mem]` state — pure two-timescale recurrence) or more powerful (a deep test-time-trained memory, Titans). The recommended default is the fast-weight matrix: it is the cheapest option that still gives a genuine associative long-term store.

---

## 2. Background: the current pipeline

**Re-verified integration anchors** (read directly from the repo):

| Component | File | Lines | Role for memory |
|---|---|---|---|
| AC predictor `forward` | `starVLA/model/modules/world_model/vj2_predictor.py` | class 17–50; `forward` 138–196; action/frame interleave 150–160; mask slice 162; block loop 165–187; output split 189–191 | **Where `[mem]` tokens are prepended and read back** |
| Frame-causal mask | `vj2_modules.py` | `build_action_block_causal_attention_mask` 12–23 | **Mask to extend for memory rows/cols** |
| RoPE attention | `vj2_modules.py` | `ACRoPEAttention` 114–263; SDPA 248–252; action-token handling | **Memory tokens must be RoPE-exempt here (the one subtle bug)** |
| Cross-attention primitive | `vj2_modules.py` | `CrossAttention` 571–599; `CrossAttentionBlock` 602–615 | Reusable compressor (slow tier, optional) |
| Framework orchestration | `starVLA/model/framework/VLA_JEPA.py` | encoder frozen `no_grad` 254–256; latent split 261–262; predictor call 265–268; `wm_loss` 270–274; capture 112–130, 277; action head 306; loss returns 280, 308 | **Where memory is carried, written, reset; anti-leakage lives here** |
| DiT action head | `model/modules/action_model/GR00T_ActionHeader.py` | head call 306; conditioning `sa_embs`/`vl_embs` 270–317 | **Optional memory-conditioned decisions** |
| Qwen action tokens | `VLA_JEPA.py` | `action_tokens` 240; `embodied_action_tokens` 241 | Token streams memory can fuse into |
| JEPA analysis | `starVLA/training/trainer_utils/jepa_analysis.py` | `compute_jepa_scalar_stats` 78–172; `make_jepa_figures` 176–278 | **Extend with memory scalars** |
| Trainers | `train_vlajepa_cotrain.py` 322–361, `_log_jepa` 381–399; `train_vlajepa_video.py` 272–295 | `capture_jepa` toggle; W&B logging; **2-pass step** | Where state is carried/detached/reset |
| Dataloader | `dataloader/gr00t_lerobot/datasets.py` 105–112, 236–239, 637–643 | `ModalityConfig.delta_indices`, windowing | **Where `episode_index`/`done` must be surfaced** |

**Current data flow:**

```text
CURRENT VLA-JEPA  (Markovian beyond the ~8-frame clip)
================================================================================
 frames [B,V,T,H,W,3]                 instruction text
        │                                    │
        ▼                                    ▼
 ┌──────────────────────┐           ┌──────────────────────┐
 │ V-JEPA2 ENCODER      │           │ Qwen3-VL             │
 │ FROZEN (no_grad)     │           │ FROZEN               │
 │ VLA_JEPA.py:254-256  │           │ VLA_JEPA.py:240-241  │
 └──────────┬───────────┘           └──────────┬───────────┘
   per-frame latents                    action_tokens / embodied_action_tokens
            │                                   │
   input_states (current obs) :261             │
   gt_states  (future target) :262 ──(stop-grad supervision)
            ▼                                   │
 ┌──────────────────────────────┐              │
 │ AC PREDICTOR                 │              │  objective 1: next-frame latent
 │ frame-causal, window T       │              │  wm_loss = L1(pred[0:T-1], sg gt[1:T])
 │ vj2_predictor.py:138-196     │              │  VLA_JEPA.py:270-274  (×0.1 joint)
 └──────────┬───────────────────┘              │
   predicted latents                           ▼
            │                       ┌──────────────────────────────┐
            └─────────────────────► │ DiT FLOW-MATCH HEAD          │ objective 2:
                                    │ VLA_JEPA.py:306              │ action chunk
                                    └──────────────┬───────────────┘
                                                   ▼
                                            OUTPUT: action chunk
================================================================================
```

Constraints the design must respect: the encoder is **frozen** under `no_grad` (`:254–256`); supervision is **leakage-free** — the predictor sees only current-frame latents while future latents are stop-gradient targets (`gt_states`, `:262`); the action head reads `embodied_action_tokens` (`:306`); `wm_loss` is downweighted ×0.1 in the joint path (`:308`).

---

## 3. The memory gap

```text
THE GAP — three things are missing, nothing more
================================================================================
 [A] HARD HORIZON CEILING  — context = frame-causal window T (~8 frames)
     local_window_time = T                              (vj2_modules.py:17-21)
 [B] NO PERSISTENT STATE    — predictor latents discarded each forward;
     nothing carried across clips or across the 2 cotrain passes
 [C] NO LONG-TERM RECALL    — no store keyed by latents; once a clip leaves
     the window, its information is gone
================================================================================
 close [A]+[B] with short-term recurrence; close [C] with an associative store.
================================================================================
```

That is the whole problem. We do **not** need cross-episode session management, a knowledge base, or a planner-with-memory. We need (i) state that survives between clips and (ii) a bounded associative memory that survives across a long episode. Two tiers, two mechanisms.

---

## 4. Design principles

Five rules; everything in §5 is a consequence of them.

1. **Fixed size.** Memory is a tensor of chosen shape, independent of episode length or number of episodes seen. No growth, therefore no eviction policy.
2. **One closed-form update rule per tier.** "What to write" and "what to forget" are learned gates, not hand-tuned thresholds or controllers.
3. **Fully differentiable.** Reads are forward passes (attention / matvec), never argmax/top-k. Gradients flow into the read/write projections from the *existing* losses — the model **learns what to remember**. (Contrast Memorizing Transformers [arXiv:2203.08913], which must *stop-grad* around its kNN bank.)
4. **Reset-aware by construction.** Episode boundaries clear memory with a single per-sample assignment — no cross-episode leakage unless explicitly wanted.
5. **Real-time and surgical.** O(1)-ish per step, identical code at train and deploy, and it **reuses existing machinery** (the predictor's add-token concat, the causal-mask builder, the action head's cross-attention). Target diff ≈ 200–300 lines; the frozen encoder and Qwen are untouched.

These are exactly the properties differentiable-memory work was built for [Neural Turing Machines, arXiv:1410.5401; DNC, Nature 2016] — minus the addressing complexity, because a fixed slate plus a delta rule already gives content-based write and forget.

---

## 5. The architecture — two tiers, two timescales

```text
ELEGANT MEMORY — two in-model tiers   (encoder + Qwen FROZEN)
================================================================================
                              ┌─────────────────────────────────────┐
   carried across clips ─────►│ AC PREDICTOR  (vj2_predictor.py)     │
   (detached, reset @episode) │                                     │
                              │  [ mem(M) | act,frame tokens ... ]  │  ← READ
   ┌──────────────────────┐   │   ▲ prepend          ▲ linear-attn  │
   │ SHORT-TERM           │   │   │ recurrent tokens  │ read m=Wᵀq   │
   │ M recurrent [mem]    │◄──┼───┘                   │             │
   │ tokens  (RMT/BRT)    │   │                       │             │
   │ h ∈ [B, M, D]        │   └───────────┬───────────┼─────────────┘
   └──────────────────────┘    predicted latents      │
                                          │            │ write (delta rule,
                                          ▼            │  current-obs only)
                                    wm_loss            │
   ┌──────────────────────┐                           ▼
   │ LONG-TERM            │   W ← α·W·(I − β k kᵀ) + β v kᵀ
   │ fast-weight matrix   │◄──────────────────────────────────────────
   │ W ∈ [B, d_k, d_v]    │   read m = Wᵀq  (→ predictor + optional action head)
   └──────────────────────┘
================================================================================
 short-term  = precise, small, rewritten every clip   (working memory)
 long-term   = associative, slowly-decaying, O(1)/step (episodic/task memory)
================================================================================
```

### 5.1 Short-term (working) memory — recurrent `[mem]` tokens

A fixed set of `M` learnable memory tokens (start `M = 8`, matching the per-step action-token count so it reuses the existing add-token path). The **predictor itself is the recurrent cell** — no new attention stack.

- **State:** a learned init `mem_init ∈ [M, D]` (`nn.Parameter`) and a detached per-sample carried buffer `h ∈ [B, M, D]`.
- **Read-in:** prepend `h` as `M` leading tokens right after the action/frame interleave (`vj2_predictor.py:160`). Every frame token attends to them.
- **Write-out:** after the predictor blocks, take the output states at the `M` mem positions, `h̃ = x[:, :M, :]` (before the cond-token strip at `:189–191`). The network learns through attention+MLP what to overwrite — RMT's "input mem tokens / output mem tokens are the same positions."
- **Gated update (one rule):** `h ← (1−g)⊙h + g⊙tanh(h̃)`, `g = σ(W_g[h; h̃])`. `g→0` keeps, `g→1` overwrites (Block-Recurrent gate; initialize the gate bias to "remember"). The plain RMT copy `h ← h̃` is the gate-free special case.
- **Carry / reset:** `h` is stored on the module and **detached** except over the last `k_bptt` clips (truncated BPTT, `k=2–4`). At an episode boundary, `h ← mem_init`.

This makes the policy non-Markovian *across clips within an episode* at near-zero cost (`M` extra rows), fully differentiable, fixed-size, no store.

### 5.2 Long-term memory — associative fast-weight matrix

A single fixed-size matrix `W ∈ [B, d_k, d_v]` (e.g. `d_k = d_v = 128` ⇒ 16K floats/sample — trivial), the **fast weights** that *are* the long-term memory [Schmidhuber 1992; arXiv:2102.11174]. One write and one read per clip (per clip, not per spatial token, for the control-loop budget).

From a current-observation summary `s` (mean-pool of the newest observed frame's tokens), small linear heads produce `k = ℓ2(W_K s)`, `v = W_V s`, query `q = ℓ2(W_Q s)`, write strength `β = σ(w_β·s) ∈ (0,1)`, decay gate `α = σ(w_α·s) ∈ (0,1)`:

- **Read (before write):** `m = Wᵀ q` — one matvec, linear attention [arXiv:2006.16236]. Project `m` to `D` and inject (§5.3).
- **Write (Gated DeltaNet [arXiv:2412.06464]):** `W ← α·W·(I − β k kᵀ) + β v kᵀ`. The Householder term `(I − β k kᵀ)` **erases the old value at key `k` before writing the new one** (targeted overwrite, no controller); `α` is uniform decay.
- **Forgetting = the gates.** `α→1` long retention, `α→0` rapid wipe; the delta term edits a single association. No eviction, no capacity check.
- **Carry / reset:** `W` is a detached runtime buffer (truncated BPTT over `k_bptt` clips); at an episode boundary, `W ← 0`.

Read = one matvec, write = one outer-product + one rank-1 projection — **O(d_k·d_v), independent of how long the episode has run.** This is the cheapest possible genuine associative memory, and it is the right fit for a robot loop where you cannot grow a KV cache or run kNN every step.

### 5.3 How memory reaches decisions

- **Predictor (primary):** the `M` short-term tokens are read natively (they are in the sequence); the long-term read is projected up (`Linear(d_v→D)`) and prepended as **one extra leading "memory-read" token**. Total leading block = `M + 1` tokens, sliced off downstream. Both then condition every next-latent prediction.
- **Action head (optional, recommended for manipulation):** concatenate the projected long-term read as one extra conditioning token in `sa_embs`/`vl_embs` (`GR00T_ActionHeader.py:302–308`), behind a config flag. This lets the policy act on episode-level context ("I already grasped object A") even when the current 8-frame clip is ambiguous — the world-model-only video pass is unaffected.

### 5.4 The anti-leakage invariant (non-negotiable)

The predictor is trained to forecast `gt_states` (future targets, `:262`) from `input_states` (current obs, `:261`). **Memory is written *only* from `input_states`** — never from `gt_states` and never from `predicted_states` (which contain the answer). Combined with **read-before-write ordering** within a clip, this guarantees the current prediction is supervised against `gt` without the model ever reading the future out of memory. (`input_states` already carries no encoder gradient — the encoder runs under `no_grad` at `:254` — so the write heads train via the read path on later clips.)

### 5.5 Reset and the one subtle bug

**Two correctness items dominate the implementation:**

1. **RoPE / reshape collision (the load-bearing detail).** Prepended memory tokens have **no grid position**, and the predictor's per-frame `x.view(B, T, cond+H*W, D)` reshapes (`:154`, `:190`) assume `seq_len = T*(cond+H*W)`. So memory tokens must (a) stay a **leading block sliced off before those reshapes**, and (b) be **exempted from spatial RoPE** in `ACRoPEAttention` — route them through the existing action-token (temporal-only / NoPE) path (`vj2_modules.py:184–206`), the template the code already uses for non-grid tokens. Get this wrong and `rotate_queries_or_keys` silently corrupts the memory.
2. **Per-sample reset.** At an episode boundary, `h ← mem_init`, `W ← 0` — applied with a per-sample `done` mask (`W ← W·(1−done)[:,None,None]`) so finishing one env in a batch does not reset the others. Inference already calls `model.reset(...)` per episode (`examples/LIBERO/eval_libero.py:144` → `model2libero_interface.py:69`); add one `reset_memory()` call alongside the existing `image_history.clear()`.

The **mask extension** is small: grow `build_action_block_causal_attention_mask` to `[(M+1)+N, (M+1)+N]`; the `M+1` leading rows/cols are `True` against the current clip (frames read memory; memory summarizes the clip), the existing frame-causal sub-block is preserved, and per-clip reset (not the mask) carries cross-clip information.

### 5.6 The design dial (pick your point on the elegance ↔ capacity curve)

The short-term tier is the same in all three. Only the **long-term** tier changes:

| Variant | Long-term mechanism | Cost / step | Recall quality | When to pick |
|---|---|---|---|---|
| **Minimal — Two-Timescale Recurrence** | a *second* `[mem]` state on a slow clock (ticks every `P` clips), consolidated from the fast state by one cross-attention + gate | cheapest; no associative store | lossy summary | maximum simplicity; "one mechanism, two clocks" [Clockwork RNN, arXiv:1402.3511; MTS3, arXiv:2310.18534; Compressive Transformer, arXiv:1911.05507] |
| **Recommended — Fast-weight matrix** | associative `W` + (gated) delta rule, read by linear attention | O(d_k·d_v), one matvec | genuine key→value recall, bounded | **default**: associative memory at trivial cost [arXiv:2412.06464; arXiv:2406.06484] |
| **Heavy — Titans neural memory** | a small deep MLP whose weights are updated online by a surprise (inner-gradient) step + momentum + adaptive decay | inner backward/step | strongest; learns-to-memorize | only if the matrix saturates [Titans, arXiv:2501.00663; TTT, arXiv:2407.04620; Miras, arXiv:2504.13173] |

All three are in-model, fixed-size, differentiable, reset by assignment. The recommended fast-weight matrix is the sweet spot: a true associative store with no inner-loop optimizer and no extra forward/backward, so it stays inside the robot control budget. (The optional working-state alternative for the short-term tier — a resettable SSM/Mamba recurrent state [S5-for-RL, arXiv:2303.03982; Mamba, arXiv:2312.00752] — is noted for completeness but the `[mem]`-token form reuses more existing code.)

**Honest scope note.** A fixed `W` (or fixed `[mem]` state) is genuine working / episodic-summary memory, **not** verbatim retrieval of an arbitrary observation many episodes back — that exact recall is the one thing the rejected kNN bank bought, at the price of being agentic and non-differentiable. If a task provably needs lossless long-horizon lookup, the Titans MLP (or a small NTM/DNC slot memory) is the heavier, still-differentiable fallback.

---

## 6. Training

**Frozen throughout:** V-JEPA2 encoder (`VLA_JEPA.py:254`, `no_grad`) and Qwen3-VL. **Trainable (new):** `mem_init`, the gate/projection heads (`W_g`, `W_K/W_V/W_Q`, `w_α/w_β`, the read projections) — all small. `h` and `W` are **runtime state, not parameters**; their *content* is produced online, their *generating heads* are trained by backprop. **No new loss is required**: the existing `teacher_forcing_wm_loss` (`:270`) and `action_loss` (`:306`) flow through the carried memory and supervise it for free — a frame the world model predicts poorly produces a large gradient through the read, the in-model analogue of "surprise."

**Prerequisite (the main pipeline lift, stated honestly).** Today the dataloader returns *shuffled, independent single clips* with no `episode_index`/`done` in the example dict (`datasets.py:723–753`), so consecutive batches are not consecutive clips. Cross-clip memory therefore needs: (a) an optional **contiguous-segment sampler** over `all_steps` grouped by trajectory, and (b) `episode_index` + a `done`/`first` flag plumbed through collate. Keep the shuffled sampler as the default fallback — under it, memory cleanly degenerates to **in-clip register tokens** (a valid Stage 0). This is the bulk of the implementation effort; the model change is small.

```text
TRAINING CURRICULUM  (each stage adds one thing; encoder + Qwen frozen)
================================================================================
 STAGE 0  sanity / no recurrence
   mem tokens + mask + RoPE-exempt path + (optional) action-head hook,
   reset-every-forward on the current shuffled loader.
   ▸ proves the memory path does not regress the Markovian baseline. (safe first PR)
        │
        ▼  enable segment sampler + episode_index/done
 STAGE 1  short-term recurrence  (video pretrain, train_vlajepa_video.py)
   carry h across clips, truncated BPTT (k=2-4), per-episode reset.
   loss = existing wm_loss, now backprop through carried h.
   ▸ working memory learned from latent surprise, action head untouched.
        │
        ▼
 STAGE 2  long-term + cotrain  (train_vlajepa_cotrain.py, 2 passes/step)
   turn on W (delta rule) in BOTH passes; writes from input_states only.
   VLA pass: action_loss + 0.1·wm_loss (unchanged) + optional action-head read.
   video pass: wm_loss. Both carry/detach/reset; per-sample done mask.
   ▸ associative episodic memory; memory-conditioned decisions.
        │
        ▼
 STAGE 3  robot finetune / eval
   stateful single-clip rollout; cache h, W on the robot; reset on env reset.
   O(1) state, O(1) per-step read — real-time. No encoder/Qwen retraining.
================================================================================
```

**Stability:** L2-normalize keys/queries (bounds `W`'s spectral radius), let `α` be the learned decay, gate-bias-init to "remember," and rely on the grad clipping already in the trainer (`train_vlajepa_cotrain.py:335`). No new optimizer.

---

## 7. Instrumentation (lightweight)

Memory health is cheap to watch and reuses the existing side-channel — **no new analysis subsystem.** Extend `last_jepa_tensors` (`VLA_JEPA.py:112–130`) with memory norms/gates and add a small `compute_memory_stats` alongside `compute_jepa_scalar_stats` (`jepa_analysis.py:78`); the trainers already route these to W&B (`_log_jepa`, `train_vlajepa_cotrain.py:381–399`).

Log, per logging step:

- `jepa/mem_gate_g`, `jepa/mem_decay_alpha`, `jepa/mem_write_beta` — are the gates alive, or collapsed to "ignore memory"?
- `jepa/mem_state_norm`, `jepa/W_fro_norm`, `jepa/mem_state_effective_rank` — drift / collapse detection.
- `jepa/mem_read_contribution_cosine` — how much the memory read moves the predicted latent.
- **Memory-ablation Δ** (the one causal probe): at a logging step, zero `h` and `W` and re-run; log the change in predicted-latent cosine and (in eval) task success. If Δ ≈ 0, memory is dead weight — surface it early.

That is the whole tooling story. The elaborate probe/counterfactual/imagined-rollout suite from the previous draft is **dropped** — it was scope creep, not part of an elegant memory design.

---

## 8. Evaluation

**Ablation ladder** (control = current Markovian VLA-JEPA):

| # | Configuration |
|---|---|
| 1 | No memory (current) |
| 2 | + short-term `[mem]` recurrence |
| 3 | + long-term fast-weight `W` |
| 4 | + action-head conditioned on the long-term read |
| 5 | long-term variant: fast-weight `W` **vs** two-timescale `[mem]` **vs** Titans MLP (§5.6) |

**Is memory actually used? (the gold standard):** the **memory-ablation Δ** of §7 at inference — zero / shuffle `h` and `W`, measure Δ task success and Δ predicted-latent cosine. A real memory effect must show a measurable drop; a flat curve means the gates collapsed.

**Where memory should help:** long-horizon / partial-observability manipulation, where information leaves the 8-frame window before it is needed. Use the existing **LIBERO / LIBERO-Plus / SimplerEnv** harness in this repo, biased toward long-horizon and multi-stage tasks; cross-check against the only *in-model* VLA memory with direct manipulation evidence, VQ-Memory [arXiv:2603.09513]. On short, fully-observable clips memory may be underused — that is expected and is itself a reported result.

| Metric | Captures | Where |
|---|---|---|
| Task success / success-by-stage | end-to-end policy quality | LIBERO / SimplerEnv |
| Success vs episode length | long-horizon memory benefit | long-horizon suites |
| Predicted–GT latent cosine | world-model fidelity | `jepa_analysis.py` (exists) |
| `eval/action_mae`, `eval/action_mse` | action quality | already wired (cotrain) |
| **Memory-ablation Δ** | **is memory causally used** | new (§7) |
| Gate / norm statistics | memory alive vs collapsed | new (§7) |
| Per-step latency; state size | real-time cost (constant by design) | deploy |

---

## 9. Roadmap, risks & honest tradeoffs

```text
ROADMAP (relative weeks) — small, sequential, each independently shippable
================================================================================
 wk 0-1  PHASE 0  mem tokens + mask + RoPE-exempt path + memory-ablation scalar,
                  reset-every-forward.  ▸ no regression vs baseline (Stage 0).
 wk 1-3  PHASE 1  segment sampler + episode_index/done; short-term recurrence,
                  truncated BPTT, per-episode reset.  ▸ ablation rows 1-2.
 wk 3-5  PHASE 2  long-term fast-weight W (gated delta rule); read into predictor.
                  ▸ ablation row 3 + memory-ablation causal test.
 wk 5-7  PHASE 3  action-head conditioning; long-horizon eval; variant sweep.
                  ▸ ablation rows 4-5; LIBERO long-horizon + SimplerEnv.
================================================================================
```

| Risk | Mitigation |
|---|---|
| **RoPE / reshape corruption** (subtlest bug) | route memory tokens through the action-token NoPE path; keep them a leading block sliced before the per-frame `view()` (§5.5). De-risk in Phase 0. |
| **Future-latent leakage** | write from `input_states` only, read-before-write; audit with the memory-ablation test (§5.4). |
| **Dataloader lift** (no `episode_index`, shuffled clips) | add a contiguous-segment sampler + `done` flag; fall back to in-clip registers until then (§6). |
| **Gate collapse** (memory ignored) | gate-bias-init to remember; monitor gate/contribution scalars; validate on tasks where memory demonstrably helps. |
| **BPTT instability / drift** | short `k_bptt` (2–4), L2-normalized keys/queries, learned `α`, existing grad clipping; per-episode reset. |
| **Fixed capacity is lossy** | accepted trade for elegance/real-time; escalate to Titans MLP only if `W` provably saturates (§5.6). |
| **Robotics fit is a proposal, not a reproduction** | RMT / DeltaNet / Titans are validated on language/long-context, not inside a frozen-V-JEPA2 VLA; every claim is gated behind the §8 ablation ladder against the Markovian baseline. |

**The single most important de-risk (Phase 0/1):** confirm that (a) adding `[mem]` tokens with the correct RoPE-exempt path does **not** regress the baseline, and (b) cheap short-term recurrence already lifts long-horizon latent cosine and success rate. If yes, the long-term fast-weight tier is justified; if not, fix the short-term tier before adding the associative store.

---

## 10. References (verified)

> All entries below were web-verified (real arXiv id / venue). Grouped by role in this design.

### 10.A — Short-term: recurrent memory in transformers
1. Bulatov, Kuratov, Burtsev. **Recurrent Memory Transformer (RMT).** NeurIPS 2022. arXiv:2207.06881. — Learnable memory tokens carried across segments; backbone unchanged. *(the short-term tier)*
2. Bulatov, Kuratov, Burtsev. **Scaling Transformer to 1M tokens and beyond with RMT.** arXiv:2304.11062.
3. Hutchins, Schlag, Wu, Dyer, Neyshabur. **Block-Recurrent Transformers.** NeurIPS 2022. arXiv:2203.07852. — LSTM-style gate on the carried state *(the gated update rule)*.
4. Dai et al. **Transformer-XL.** ACL 2019. arXiv:1901.02860. — Segment-level recurrence.
5. Wu, Lan, Qian, Gu, Geramifard, Yu. **Memformer.** Findings of AACL-IJCNLP 2022. arXiv:2010.06891.

### 10.B — Long-term: associative fast-weight memory (recommended tier)
6. Schmidhuber. **Learning to Control Fast-Weight Memories.** Neural Computation 4(1):131–139, 1992. *(origin of fast weights; not on arXiv)*.
7. Schlag, Irie, Schmidhuber. **Linear Transformers Are Secretly Fast Weight Programmers.** ICML 2021. arXiv:2102.11174.
8. Katharopoulos, Vyas, Pappas, Fleuret. **Transformers are RNNs: Linear Attention.** ICML 2020. arXiv:2006.16236. — The linear-attention read `m = Wᵀq`.
9. Yang, Wang, Shen, Panda, Kim. **Parallelizing Linear Transformers with the Delta Rule (DeltaNet).** NeurIPS 2024. arXiv:2406.06484. — Targeted overwrite write.
10. Yang, Kautz, Hatamizadeh. **Gated Delta Networks: Improving Mamba2 with the Delta Rule (Gated DeltaNet).** ICLR 2025. arXiv:2412.06464. — `W ← α·W·(I − β k kᵀ) + β v kᵀ` *(the long-term write rule)*.
11. Yang, Zhang, Hua, et al. **Gated Linear Attention (GLA).** ICML 2024. arXiv:2312.06635.
12. Ramsauer et al. **Hopfield Networks is All You Need.** ICLR 2021. arXiv:2008.02217. — Attention as associative memory.
13. Wang, Shi, Fox. **Test-time Regression: a Unifying Framework for Sequence Models.** arXiv:2501.12352. — Associative-memory view of modern recurrences.

### 10.C — The minimal variant: two-timescale / hierarchical recurrence
14. Koutník, Greff, Gomez, Schmidhuber. **A Clockwork RNN.** ICML 2014. arXiv:1402.3511.
15. Chung, Ahn, Bengio. **Hierarchical Multiscale RNN.** ICLR 2017. arXiv:1609.01704.
16. Rae, Potapenko, Jayakumar, Lillicrap. **Compressive Transformers.** ICLR 2020. arXiv:1911.05507. — Consolidation by compression.
17. Shaj et al. **Multi Time Scale World Models (MTS3).** NeurIPS 2023 (Spotlight). arXiv:2310.18534.
18. Hwang, Wang, Gu. **Dynamic Chunking for Hierarchical Sequence Modeling (H-Net).** arXiv:2507.07955. — Optional learned boundary clock.

### 10.D — The heavy variant: test-time-trained neural memory
19. Sun et al. **Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT).** ICML 2025. arXiv:2407.04620.
20. Behrouz, Zhong, Mirrokni. **Titans: Learning to Memorize at Test Time.** arXiv:2501.00663. — Deep memory + surprise + momentum + adaptive decay.
21. Behrouz et al. **It's All Connected (Miras).** arXiv:2504.13173. — The 4-knob generalization.
22. Behrouz et al. **ATLAS: Learning to Optimally Memorize the Context at Test Time.** arXiv:2505.23735.

### 10.E — Recurrent-state / SSM working memory (alternative short-term carrier)
23. Gu, Goel, Ré. **Structured State Spaces (S4).** ICLR 2022. arXiv:2111.00396.
24. Smith, Warrington, Linderman. **Simplified State Space Layers (S5).** ICLR 2023. arXiv:2208.04933.
25. Lu et al. **Structured State Space Models for In-Context RL (S5-for-RL).** NeurIPS 2023. arXiv:2303.03982. — The `(1 − done)` resettable-state trick.
26. Gu, Dao. **Mamba.** COLM 2024. arXiv:2312.00752.
27. Dao, Gu. **Transformers are SSMs (Mamba-2).** ICML 2024. arXiv:2405.21060.
28. Ota. **Decision Mamba.** arXiv:2403.19925.

### 10.F — Differentiable-memory ancestry & the agentic baseline we reject
29. Graves, Wayne, Danihelka. **Neural Turing Machines.** arXiv:1410.5401.
30. Graves et al. **Hybrid computing with dynamic external memory (DNC).** Nature 538:471–476, 2016.
31. Wu, Rabe, Hutchins, Szegedy. **Memorizing Transformers.** ICLR 2022. arXiv:2203.08913. — kNN memory with a *stop-grad* gate (the non-differentiability we avoid).
32. Packer et al. **MemGPT: LLMs as Operating Systems.** arXiv:2310.08560. *(agentic paging — rejected)*.
33. Lewis et al. **Retrieval-Augmented Generation (RAG).** NeurIPS 2020. arXiv:2005.11401. *(external retrieval — rejected)*.
34. Paischer et al. **History Compression via Language Models in RL (HELM).** ICML 2022. arXiv:2205.12258.

### 10.G — Memory in VLA / robot world models (context & cross-checks)
35. Shi et al. **MemoryVLA.** arXiv:2508.19236. *(perceptual-cognitive memory bank — agentic camp)*.
36. **VQ-Memory: Robust Long-Horizon Manipulation in Non-Markovian Benchmarks.** arXiv:2603.09513. — In-model VQ memory; the elegant-camp manipulation cross-check.
37. **EchoVLA: Synergistic Declarative Memory for VLA Mobile Manipulation.** arXiv:2511.18112.
38. Li, Guo, Wu, et al. **MAP-VLA: Memory-Augmented Prompting for VLA.** arXiv:2511.09516.

### 10.H — World-model / VLA substrate
39. Assran, Bardes, et al. (Meta FAIR). **V-JEPA 2.** arXiv:2506.09985.
40. Sun et al. **VLA-JEPA: Enhancing VLA with a Latent World Model.** arXiv:2602.10098.
41. Hafner, Pasukonis, Ba, Lillicrap. **DreamerV3.** Nature 640:647–653, 2025. arXiv:2301.04104. — RSSM recurrent latent state.
42. Hafner et al. **Learning Latent Dynamics for Planning from Pixels (PlaNet / RSSM).** ICML 2019. arXiv:1811.04551.
43. Hansen, Su, Wang. **TD-MPC2.** ICLR 2024. arXiv:2310.16828.
44. Wu et al. **iVideoGPT.** NeurIPS 2024. arXiv:2405.15223.
45. Zhou, Pan, LeCun, Pinto. **DINO-WM.** ICML 2025. arXiv:2411.04983.

---

*End of proposal. Descriptive only; implementing it requires the code changes outlined per `file:line` above, none of which have been made. The headline design (recurrent `[mem]` tokens + fast-weight associative matrix) is the recommended default; §5.6 gives a minimal and a heavy variant on the same in-model, differentiable, reset-aware foundation.*
