"""memv3.1 warm-start surgery: competent backbone + reading memory stack.

Base  = the no-memory 100K co-train (LIBERO 47/78/88/91) — supplies Qwen,
        encoder, and the competent action head.
Donor = the memv3 M2 40K live arm — supplies the memory stack whose read the
        gate ladder certified (gap_act +0.143, control 0) AND the
        retrodiction-capable shared predictor the retro losses require.

  python scripts/analysis/m3p1_merge_warmstart.py --out /abs/merged.pt
"""

import argparse

import torch

BASE = (
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/"
    "vlajepa_cotrain_allv2/checkpoints/VLA-JEPA-allv2-step_100000.pt"
)
DONOR = (
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/"
    "vlajepa_m3_retro_m2/checkpoints/VLA-JEPA-m3-live-step_40000.pt"
)
DONOR_PREFIXES = (
    "memory_module.",
    "retro_cond_proj.",
    "retro_pick_head.",
    "memory_read_proj.",
    "wm_mask_token",
    "vj_predictor.",  # retrodiction lives in the shared predictor weights
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--donor", default=DONOR)
    args = parser.parse_args()

    merged = torch.load(args.base, map_location="cpu")
    donor = torch.load(args.donor, map_location="cpu")
    stale = [k for k in merged if k.startswith(DONOR_PREFIXES)]
    for key in stale:
        del merged[key]
    moved = 0
    for key, value in donor.items():
        if key.startswith(DONOR_PREFIXES):
            merged[key] = value
            moved += 1
    if moved == 0:
        raise RuntimeError("donor contributed no memory keys — wrong checkpoint?")
    torch.save(merged, args.out)
    print(f"merged: {len(merged)} tensors ({moved} from donor, {len(stale)} base keys replaced)")


if __name__ == "__main__":
    main()
