# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
JEPA latent world-model representation analysis.

The VLA-JEPA world model (``VisionTransformerPredictorAC``) predicts the V-JEPA2
latent of the *next* frame from the latents of the *current* frame plus action
tokens. The quality of this latent prediction -- and whether the representation
collapses -- is the core scientific signal of JEPA pretraining. This module turns
the raw predictor tensors into a rich set of scalar metrics and matplotlib figures
suitable for W&B logging.

All functions are defensive: any failure on an individual metric is swallowed and
that metric is simply omitted, so analysis never crashes training. Inputs are
expected to be detached tensors of shape ``[B, N_tokens, D]`` (D is the multi-view
concatenated V-JEPA2 latent dim, i.e. 2 * encoder_hidden for two camera views).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, N, D] (or [B*N, D]) -> [B*N, D] float32 on the original device."""
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    return x.detach().float()


def _subsample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    n = x.shape[0]
    if n <= max_rows:
        return x
    idx = torch.randperm(n, device=x.device)[:max_rows]
    return x[idx]


def _singular_values(x: torch.Tensor, max_rows: int = 4096) -> Optional[torch.Tensor]:
    """Centered singular values of a [N, D] feature matrix (collapse spectrum)."""
    try:
        x = _subsample_rows(x, max_rows)
        x = x - x.mean(dim=0, keepdim=True)
        # svdvals is cheap for D~2048; guard against numerical issues.
        return torch.linalg.svdvals(x)
    except Exception:
        return None


def _effective_rank(svals: torch.Tensor) -> Dict[str, float]:
    """Two standard effective-rank measures from singular values.

    - participation ratio:  (sum s^2)^2 / sum s^4   (a.k.a. stable rank-ish, dim-like)
    - entropy effective rank: exp(H(p)), p = s / sum s   (Roy & Vetterli, 2007)
    Both shrink toward 1 as the representation collapses to a single direction.
    """
    out: Dict[str, float] = {}
    try:
        s = svals[svals > 0]
        if s.numel() == 0:
            return out
        s2 = s * s
        pr = (s2.sum() ** 2) / (s2 * s2).sum().clamp_min(1e-12)
        out["participation_ratio"] = float(pr)
        p = s / s.sum().clamp_min(1e-12)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum()
        out["effective_rank"] = float(torch.exp(ent))
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------- scalars
@torch.no_grad()
def compute_jepa_scalar_stats(
    predicted: torch.Tensor,
    gt: torch.Tensor,
    input_states: Optional[torch.Tensor] = None,
    action_tokens: Optional[torch.Tensor] = None,
    num_views: int = 2,
    rank_max_rows: int = 2048,
) -> Dict[str, float]:
    """Cheap-ish per-step JEPA representation metrics (safe to call every log step).

    Returns a flat dict of float scalars (no tensors). Keys are namespaced for W&B
    panel grouping (``jepa/...`` and ``jepa_view/...``).
    """
    stats: Dict[str, float] = {}
    try:
        pred = predicted.detach().float()
        tgt = gt.detach().float()
        if pred.shape != tgt.shape:
            # Align on the common token length (defensive against off-by-one wiring).
            n = min(pred.shape[1], tgt.shape[1])
            pred, tgt = pred[:, :n], tgt[:, :n]

        # --- prediction fidelity ---
        cos = F.cosine_similarity(pred, tgt, dim=-1)  # [B, N]
        stats["jepa/pred_gt_cosine_mean"] = float(cos.mean())
        stats["jepa/pred_gt_cosine_std"] = float(cos.std())
        stats["jepa/pred_gt_cosine_p10"] = float(cos.flatten().quantile(0.10))
        stats["jepa/frac_tokens_cos_gt_0.9"] = float((cos > 0.9).float().mean())
        stats["jepa/pred_gt_l1"] = float((pred - tgt).abs().mean())
        stats["jepa/pred_gt_l2"] = float((pred - tgt).pow(2).mean().sqrt())

        # --- representation magnitude / collapse (feature std across tokens) ---
        pf = _flatten_tokens(pred)
        tf = _flatten_tokens(tgt)
        pred_feat_std = float(pf.std(dim=0).mean())
        gt_feat_std = float(tf.std(dim=0).mean())
        stats["jepa/pred_feature_std"] = pred_feat_std
        stats["jepa/gt_feature_std"] = gt_feat_std
        stats["jepa/feature_std_ratio"] = pred_feat_std / max(gt_feat_std, 1e-8)
        # variance across tokens within a sample (per-sample diversity); low => collapse
        stats["jepa/pred_token_variance"] = float(pred.var(dim=1).mean())
        stats["jepa/gt_token_variance"] = float(tgt.var(dim=1).mean())
        stats["jepa/pred_token_norm"] = float(pred.norm(dim=-1).mean())
        stats["jepa/gt_token_norm"] = float(tgt.norm(dim=-1).mean())

        # --- effective rank (subsampled, every log step is fine for D~2048) ---
        sp = _singular_values(pf, rank_max_rows)
        sg = _singular_values(tf, rank_max_rows)
        if sp is not None:
            for k, v in _effective_rank(sp).items():
                stats[f"jepa/pred_{k}"] = v
        if sg is not None:
            for k, v in _effective_rank(sg).items():
                stats[f"jepa/gt_{k}"] = v
        if sp is not None and sg is not None:
            er_p = stats.get("jepa/pred_effective_rank")
            er_g = stats.get("jepa/gt_effective_rank")
            if er_p and er_g:
                stats["jepa/effective_rank_ratio"] = er_p / max(er_g, 1e-8)

        # --- identity baseline: does the predictor beat "copy current frame"? ---
        if input_states is not None:
            inp = input_states.detach().float()
            n = min(inp.shape[1], tgt.shape[1])
            base_cos = F.cosine_similarity(inp[:, :n], tgt[:, :n], dim=-1)
            stats["jepa/identity_baseline_cosine"] = float(base_cos.mean())
            stats["jepa/pred_gain_over_identity"] = float(cos.mean() - base_cos.mean())
            stats["jepa/input_token_norm"] = float(inp.norm(dim=-1).mean())

        # --- action-conditioning magnitude proxy ---
        if action_tokens is not None:
            at = action_tokens.detach().float()
            a_norm = float(at.norm(dim=-1).mean())
            stats["jepa/action_token_norm"] = a_norm
            if input_states is not None:
                stats["jepa/action_to_state_norm_ratio"] = a_norm / max(
                    stats.get("jepa/input_token_norm", 1.0), 1e-8
                )

        # --- per-view split (the latent dim concatenates num_views camera latents) ---
        D = pred.shape[-1]
        if num_views and num_views > 1 and D % num_views == 0:
            vd = D // num_views
            for v in range(num_views):
                pv = pred[..., v * vd : (v + 1) * vd]
                gv = tgt[..., v * vd : (v + 1) * vd]
                vc = F.cosine_similarity(pv, gv, dim=-1).mean()
                stats[f"jepa_view/view{v}_cosine"] = float(vc)
                stats[f"jepa_view/view{v}_pred_std"] = float(
                    _flatten_tokens(pv).std(dim=0).mean()
                )
    except Exception:
        pass
    return stats


# ----------------------------------------------------------------------------- figures
@torch.no_grad()
def make_jepa_figures(
    predicted: torch.Tensor,
    gt: torch.Tensor,
    input_states: Optional[torch.Tensor] = None,
    max_rows: int = 2048,
) -> Dict[str, "object"]:
    """Build matplotlib figures visualizing JEPA representation quality/collapse.

    Returns {name: Figure}. Caller is responsible for wandb.Image wrapping + closing.
    """
    figs: Dict[str, object] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return figs

    try:
        pred = predicted.detach().float()
        tgt = gt.detach().float()
        n = min(pred.shape[1], tgt.shape[1])
        pred, tgt = pred[:, :n], tgt[:, :n]
        pf = _subsample_rows(_flatten_tokens(pred), max_rows)
        tf = _subsample_rows(_flatten_tokens(tgt), max_rows)

        # (1) Singular-value spectrum (log) -- the canonical collapse diagnostic.
        try:
            sp = _singular_values(pf, max_rows)
            sg = _singular_values(tf, max_rows)
            if sp is not None and sg is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot((sp / sp.max()).cpu().numpy(), label="predicted", lw=2)
                ax.plot((sg / sg.max()).cpu().numpy(), label="ground-truth", lw=2)
                ax.set_yscale("log")
                ax.set_xlabel("singular value index")
                ax.set_ylabel("normalized singular value")
                ax.set_title("JEPA latent spectrum (flat = healthy, steep = collapse)")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()
                figs["spectrum"] = fig
        except Exception:
            pass

        # (2) Per-token cosine similarity histogram.
        try:
            cos = F.cosine_similarity(pred, tgt, dim=-1).flatten().cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(cos, bins=60, range=(-1, 1), color="#9C276A", alpha=0.85)
            ax.axvline(float(cos.mean()), color="k", ls="--", label=f"mean={cos.mean():.3f}")
            ax.set_xlabel("cosine(predicted, ground-truth) per token")
            ax.set_ylabel("count")
            ax.set_title("Predicted-vs-GT latent cosine similarity")
            ax.legend()
            fig.tight_layout()
            figs["cosine_hist"] = fig
        except Exception:
            pass

        # (3) Sorted per-dimension std: predicted vs gt (collapse => many near-zero dims).
        try:
            ps = pf.std(dim=0).sort(descending=True).values.cpu().numpy()
            gs = tf.std(dim=0).sort(descending=True).values.cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ps, label="predicted", lw=1.5)
            ax.plot(gs, label="ground-truth", lw=1.5)
            ax.set_xlabel("feature dimension (sorted by std)")
            ax.set_ylabel("per-dim std")
            ax.set_title("Per-dimension activation std")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            figs["dim_std"] = fig
        except Exception:
            pass

        # (4) 2D PCA scatter: project GT + predicted onto GT's top-2 principal axes.
        try:
            mu = tf.mean(dim=0, keepdim=True)
            tc = tf - mu
            # top-2 right singular vectors of GT
            _, _, vh = torch.linalg.svd(_subsample_rows(tc, 1024), full_matrices=False)
            comp = vh[:2].T  # [D, 2]
            gt2 = (tc @ comp).cpu().numpy()
            pc = (pf - mu) @ comp
            pc = pc.cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(gt2[:, 0], gt2[:, 1], s=4, alpha=0.4, label="ground-truth")
            ax.scatter(pc[:, 0], pc[:, 1], s=4, alpha=0.4, label="predicted")
            ax.set_xlabel("PC1 (GT basis)")
            ax.set_ylabel("PC2 (GT basis)")
            ax.set_title("Latent token cloud (PCA on GT basis)")
            ax.legend()
            fig.tight_layout()
            figs["pca_scatter"] = fig
        except Exception:
            pass
    except Exception:
        pass
    return figs
