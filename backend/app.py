"""Private, bounded audio upload API. Run from repository root."""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

logger = logging.getLogger("voxguard")
MAX_BYTES = int(os.getenv("VOXGUARD_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_DURATION = float(os.getenv("VOXGUARD_MAX_DURATION_SECONDS", "120"))
MIN_DURATION = 0.5
FORMATS = {".wav": {"wav"}, ".mp3": {"mp3"}, ".m4a": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}, ".flac": {"flac"}}
_gate = threading.Lock()
_model = None


class BodyLimitMiddleware:
    """Bound multipart parsing even for chunked uploads; never log the body."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        limit = MAX_BYTES + 65536  # bounded multipart envelope
        data = bytearray()
        try:
            while True:
                event = await asyncio.wait_for(receive(), timeout=30)
                if event["type"] == "http.disconnect":
                    return
                data.extend(event.get("body", b""))
                if len(data) > limit:
                    return await JSONResponse({"detail": "Upload exceeds the configured size limit."}, 413)(scope, receive, send)
                if not event.get("more_body", False):
                    break
        except TimeoutError:
            return await JSONResponse({"detail": "Upload timed out."}, 408)(scope, receive, send)
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(data), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


app = FastAPI(title="VoxGuard AI", version="0.1.0")
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=[s.strip() for s in os.getenv("VOXGUARD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if s.strip()], allow_methods=["GET", "POST"], allow_headers=["Content-Type"], allow_credentials=False)


def get_model():
    global _model
    if _model is None:
        module = importlib.import_module("ml.service")
        _model = module.ModelService()
    return _model


def model_info():
    try:
        return get_model().info()
    except Exception:
        return {"ready": False, "name": "Anti-spoofing model", "status": "unavailable", "detail": "Model unavailable. Follow the model setup instructions."}


@app.get("/api/health")
def health():
    info = model_info()
    return {"status": "ok", "model": info, "ffmpeg_available": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "limits": {"max_upload_bytes": MAX_BYTES, "max_duration_seconds": MAX_DURATION, "min_duration_seconds": MIN_DURATION}}


@app.get("/api/model/info")
def info():
    return model_info()


def decode_audio(path: Path, suffix: str, output: Path):
    """Use content probing and bounded decoding without a shell or network protocols."""
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_format", "-show_streams", "-of", "json", str(path)], capture_output=True, timeout=15, check=True)
        metadata = json.loads(probe.stdout)
        streams = metadata.get("streams", [])
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        if len(audio) != 1 or any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic") for s in streams):
            raise HTTPException(422, "Upload a recording with one audio stream and no video.")
        detected = set(metadata.get("format", {}).get("format_name", "").split(","))
        if not detected & FORMATS[suffix]:
            raise HTTPException(415, "File content does not match its supported audio extension.")
        declared = float(metadata.get("format", {}).get("duration", "0"))
        if not np.isfinite(declared) or declared > MAX_DURATION:
            raise HTTPException(422, "Recording exceeds the configured duration limit.")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(path), "-map", "0:a:0", "-t", str(MAX_DURATION + 0.1), "-ac", "1", "-ar", "16000", "-f", "f32le", "-y", str(output)], capture_output=True, timeout=45, check=True)
        samples = np.fromfile(output, dtype="<f4")
        duration = len(samples) / 16000
        if duration > MAX_DURATION:
            raise HTTPException(422, "Recording exceeds the configured duration limit.")
        if duration < MIN_DURATION:
            raise HTTPException(422, "Recording must contain at least 0.5 seconds of audio.")
        if not np.isfinite(samples).all():
            raise HTTPException(422, "Recording contains invalid audio samples.")
        if float(np.max(np.abs(samples))) < 1e-6:
            raise HTTPException(422, "Recording is silent. Upload audible speech.")
        return samples, {"original_format": metadata["format"]["format_name"], "channels": int(audio[0].get("channels", 1)), "sample_rate": int(audio[0].get("sample_rate", 16000)), "output_channels": 1, "output_sample_rate": 16000}
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(503, "FFmpeg and FFprobe must be installed on the server.") from None
    except (subprocess.SubprocessError, ValueError, KeyError, OSError):
        raise HTTPException(422, "Audio could not be decoded. Upload a valid WAV, MP3, M4A, or FLAC recording.") from None


def analyze_path(path, suffix, output):
    samples, processing = decode_audio(path, suffix, output)
    started = time.perf_counter()
    try:
        result = get_model().analyze(samples, sample_rate=16000)
    except Exception as exc:
        if type(exc).__name__ == "ModelUnavailable" or isinstance(exc, ImportError):
            raise HTTPException(503, "Detection model unavailable. Follow the model setup instructions.") from None
        logger.error("Model inference failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Analysis could not be completed. Check server model configuration.") from None
    return {**result, "duration_seconds": len(samples) / 16000, "sample_rate": 16000, "processing": processing, "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in FORMATS:
        await file.close()
        raise HTTPException(415, "Supported formats: WAV, MP3, M4A, FLAC.")
    if not _gate.acquire(blocking=False):
        await file.close()
        raise HTTPException(429, "Another recording is being analyzed. Please retry shortly.")
    try:
        with tempfile.TemporaryDirectory(prefix="voxguard-") as directory:
            path = Path(directory) / ("upload" + suffix)
            total = 0
            with path.open("wb") as handle:
                while chunk := await file.read(65536):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise HTTPException(413, "Upload exceeds the configured size limit.")
                    handle.write(chunk)
            if total == 0:
                raise HTTPException(422, "The uploaded recording is empty.")
            return await run_in_threadpool(analyze_path, path, suffix, Path(directory) / "decoded.raw")
    finally:
        await file.close()
        _gate.release()
