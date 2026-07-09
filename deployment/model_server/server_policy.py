# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import (
    MemoryServerConfig,
    WebsocketPolicyServer,
)
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    device = torch.device(f"cuda:{str(args.cuda)}")

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()
    memory_config = MemoryServerConfig.from_env()
    if getattr(vla, "memory_enabled", False):
        # Runtime state and memory/fusion math are intentionally FP32 even when
        # the heavy backbone is served in BF16.
        vla.memory_module.float()
        vla.memory_module.capture_diagnostics = True
        # Schema 3 (memv3) has no fusion module: the read is native attention.
        fusion = getattr(vla, "policy_memory_fusion", None)
        if fusion is not None:
            # Serving always reports per-decision diagnostics; both the
            # write-side (memory_module) and read-side (fusion) captures are
            # gated behind these flags on the training path.
            fusion.float()
            fusion.capture_diagnostics = True
            if memory_config.gate_scale is not None:
                fusion.residual_scale = memory_config.gate_scale
                logging.info("MEMORY_GATE_SCALE=%s applied to fusion.residual_scale", memory_config.gate_scale)
        # Schema-3 read/cond projections stay in model dtype: predict_action
        # casts read tokens to the consumer dtype before projecting.

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host=args.host,
        port=args.port,
        metadata={"env": "simpler_env"},
        memory_config=memory_config,
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
