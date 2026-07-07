import os, sys, torch, yaml
from safetensors.torch import load_file
run, step, out_name = sys.argv[1], sys.argv[2], sys.argv[3]
src = os.path.join(run, "checkpoints", step, "model.safetensors")
sd = load_file(src)
LM = "qwen_vl_interface.model.lm_head.weight"
if LM not in sd:
    c = [k for k in sd if k.endswith("language_model.embed_tokens.weight")]
    if c: sd[LM] = sd[c[0]].clone()
out = os.path.join(run, "checkpoints", out_name)
tmp = out + f".tmp.{os.getpid()}"
with open(tmp, "wb") as h: torch.save(sd, h); h.flush(); os.fsync(h.fileno())
os.replace(tmp, out)
print("DONE", out, f"{os.path.getsize(out)/1e9:.2f} GB")
