#!/usr/bin/env python
"""Export an Accelerate full-state checkpoint (step_N/model.safetensors) into the
single weights .pt that baseframework.from_pretrained / server_policy.py expect.

The output .pt is placed at <run_dir>/checkpoints/<name>.pt so that read_mode_config
finds the sibling <run_dir>/config.yaml and <run_dir>/dataset_statistics.json.

Usage:
  python cluster/export_vlajepa_ckpt.py <run_dir> [step_dir_name] [out_name]
    run_dir        e.g. /lustre/.../vlajepa_runs/vlajepa_cotrain
    step_dir_name  e.g. step_2000  (default: read checkpoints/latest.txt)
    out_name       e.g. VLA-JEPA-cotrain-step2000.pt (default derived from step)
"""
import os
import sys
from safetensors.torch import load_file
import torch


def main():
    run_dir = sys.argv[1].rstrip("/")
    ckpt_root = os.path.join(run_dir, "checkpoints")
    step = sys.argv[2] if len(sys.argv) > 2 else None
    if step is None:
        with open(os.path.join(ckpt_root, "latest.txt")) as f:
            step = f.read().strip()
    src = os.path.join(ckpt_root, step, "model.safetensors")
    assert os.path.isfile(src), f"missing {src}"
    out_name = sys.argv[3] if len(sys.argv) > 3 else f"VLA-JEPA-{os.path.basename(run_dir)}-{step}.pt"
    out = os.path.join(ckpt_root, out_name)

    print(f"[export] loading {src}")
    sd = load_file(src)

    # safetensors cannot store shared tensors, so accelerate's save_state drops the
    # tied lm_head (Qwen3-VL ties lm_head <- input embeddings). The freshly-built model
    # lists lm_head in its state_dict, so reconstruct the tie for a strict load.
    LM_HEAD = "qwen_vl_interface.model.lm_head.weight"
    if LM_HEAD not in sd:
        cands = [k for k in sd if k.endswith("language_model.embed_tokens.weight")] or \
                [k for k in sd if k.endswith("embed_tokens.weight") and "qwen" in k.lower()]
        if cands:
            sd[LM_HEAD] = sd[cands[0]].clone()
            print(f"[export] reconstructed tied {LM_HEAD} <- {cands[0]}")
        else:
            print(f"[export] WARNING: could not find embed_tokens to tie {LM_HEAD}")

    print(f"[export] {len(sd)} tensors -> {out}")
    torch.save(sd, out)
    print(f"[export] DONE: {out}  ({os.path.getsize(out)/1e9:.2f} GB)")
    print(f"[export] eval with:  --ckpt_path {out}")


if __name__ == "__main__":
    main()
