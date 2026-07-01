# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
from dataclasses import dataclass

import torch

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy
from . import image_tools


@dataclass
class _ConnectionSession:
    """Private activation and RNG state owned by one WebSocket handler."""

    memory_state: object = None
    generator: torch.Generator | None = None
    episode_id: str | None = None
    batch_size: int | None = None
    ready: bool = False

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        memory_mode: str = "live",
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        if memory_mode not in {"live", "zero"}:
            raise ValueError(f"unsupported memory_mode: {memory_mode}")
        self._memory_mode = memory_mode
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=256 * 1024 * 1024,
            max_queue=4,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        session = _ConnectionSession()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                ret = self._route_message(msg, session=session)  # route message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logging.exception("Unexpected WebSocket handler failure")
                await websocket.send("Internal server error")
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _policy_device(self) -> torch.device:
        try:
            return next(self._policy.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _route_message(self, msg: dict, session: _ConnectionSession | None = None) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Always returns a dict containing:
            {
              "status": "ok" | "error",
              "ok": bool,
              "type": <str>,
              "request_id": <str>,
              ... (data | error)
            }
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        if not isinstance(msg, dict):
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": "default",
                "error": {"message": "Request must be a dict"},
            }
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")          # default = infer
        payload = msg.get("payload", msg)         # when no explicit payload, treat top-level as payload
        if mtype == "infer" and msg.get("reset") is True:
            # Compatibility with the original flat reset request.
            mtype = "reset"
            payload = {
                "episode_id": msg.get("instruction", "legacy"),
                "episode_seed": msg.get("episode_seed", 0),
            }
        if session is None:
            session = _ConnectionSession()

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        if mtype == "reset":
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_result",
                    "request_id": req_id,
                    "error": {"message": "Reset payload must be a dict"},
                }
            try:
                episode_seed = int(payload.get("episode_seed", 0))
                episode_id = str(payload.get("episode_id", "default"))
                device = self._policy_device()
                candidate_generator = torch.Generator(device=device)
                candidate_generator.manual_seed(episode_seed)
                # Commit reset state only after every validation/initialization
                # operation above has succeeded.
                session.generator = candidate_generator
                session.memory_state = None
                session.episode_id = episode_id
                session.batch_size = None
                session.ready = True
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "reset_result",
                    "request_id": req_id,
                    "data": {"episode_id": episode_id, "episode_seed": episode_seed},
                }
            except Exception as exc:
                return {
                    "status": "error",
                    "ok": False,
                    "type": "reset_result",
                    "request_id": req_id,
                    "error": {"message": str(exc)},
                }

        # infer
        elif mtype == "infer":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))}
                }
            try:
                memory_enabled = bool(getattr(self._policy, "memory_enabled", False))
                if memory_enabled and not session.ready:
                    raise RuntimeError("memory-enabled inference requires an explicit reset")
                batch_images = payload.get("batch_images")
                if memory_enabled and len(batch_images) != 1:
                    raise ValueError("Phase-1 serving supports exactly one episode row per connection")
                if memory_enabled:
                    if session.batch_size is None:
                        session.batch_size = len(batch_images)
                    elif session.batch_size != len(batch_images):
                        raise ValueError("batch size changed inside a live session; reset first")

                call_payload = dict(payload)
                call_payload["batch_images"] = image_tools.to_pil_preserve(batch_images)
                rng_before = (
                    session.generator.get_state().clone()
                    if session.generator is not None
                    else None
                )
                state_before = session.memory_state
                if memory_enabled:
                    call_payload.update(
                        memory_state=None if self._memory_mode == "zero" else state_before,
                        return_memory_state=True,
                        update_memory=self._memory_mode == "live",
                        generator=session.generator,
                    )
                    output_dict, candidate_state = self._policy.predict_action(**call_payload)
                    if self._memory_mode == "live":
                        session.memory_state = candidate_state
                else:
                    if session.generator is not None:
                        call_payload["generator"] = session.generator
                    output_dict = self._policy.predict_action(**call_payload)
            except Exception as e:
                if session.generator is not None and "rng_before" in locals() and rng_before is not None:
                    session.generator.set_state(rng_before)
                if "state_before" in locals():
                    session.memory_state = state_before
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)
                
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                        # "traceback": traceback.format_exc(),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
