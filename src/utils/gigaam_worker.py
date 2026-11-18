#!/usr/bin/env python3
"""Persistent JSON-lines worker for GigaAM v3 inference.

The Electron main process owns this worker and sends base64-encoded 16 kHz mono
WAV payloads over stdin. Protocol messages are written to stdout; third-party
library output is redirected to stderr so it cannot corrupt the protocol.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_model(model_name: str, requested_device: str) -> tuple[Any, str, str | None]:
    with contextlib.redirect_stdout(sys.stderr):
        import gigaam
        import torch

        device = requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            return gigaam.load_model(model_name, device=device), device, None
        except Exception as error:
            error_text = str(error).lower()
            is_cuda_error = isinstance(error, torch.cuda.OutOfMemoryError) or any(
                marker in error_text
                for marker in (
                    "cuda",
                    "cublas",
                    "cudnn",
                    "device-side",
                    "out of memory",
                )
            )
            if device != "cuda" or not is_cuda_error:
                raise

            fallback_reason = f"CUDA initialization failed: {error}"
            print(f"[GigaAM] {fallback_reason}; retrying on CPU", file=sys.stderr)
            torch.cuda.empty_cache()
            return gigaam.load_model(model_name, device="cpu"), "cpu", fallback_reason


def transcribe(model: Any, wav_bytes: bytes) -> str:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_file.write(wav_bytes)
            temp_path = audio_file.name

        with contextlib.redirect_stdout(sys.stderr):
            result = model.transcribe(temp_path)

        if hasattr(result, "text"):
            result = result.text
        return str(result or "").strip()
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def check_installation() -> int:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import gigaam  # noqa: F401
    except Exception as error:  # pragma: no cover - depends on host environment
        emit({"type": "check", "ok": False, "error": str(error)})
        return 1

    emit({"type": "check", "ok": True, "python": sys.version.split()[0]})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        return check_installation()

    try:
        load_started_at = time.perf_counter()
        model, device, fallback_reason = load_model(args.model, args.device)
        load_elapsed_ms = round((time.perf_counter() - load_started_at) * 1000)
    except Exception as error:  # pragma: no cover - depends on model download/runtime
        emit({"type": "fatal", "error": f"Unable to load GigaAM model {args.model}: {error}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    emit(
        {
            "type": "ready",
            "model": args.model,
            "device": device,
            "fallbackReason": fallback_reason,
            "loadElapsedMs": load_elapsed_ms,
        }
    )

    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("type") == "shutdown":
                emit({"type": "stopped"})
                return 0
            if request.get("type") != "transcribe" or not isinstance(request_id, str):
                raise ValueError("Invalid worker request")

            wav_base64 = request.get("wav")
            if not isinstance(wav_base64, str):
                raise ValueError("Missing WAV payload")
            wav_bytes = base64.b64decode(wav_base64, validate=True)
            if len(wav_bytes) < 44:
                raise ValueError("WAV payload is too small")

            transcribe_started_at = time.perf_counter()
            text = transcribe(model, wav_bytes)
            elapsed_ms = round((time.perf_counter() - transcribe_started_at) * 1000)
            emit({"type": "result", "id": request_id, "text": text, "elapsedMs": elapsed_ms})
        except Exception as error:
            emit({"type": "error", "id": request.get("id") if isinstance(request, dict) else None, "error": str(error)})
            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
