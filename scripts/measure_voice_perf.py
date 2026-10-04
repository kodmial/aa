#!/usr/bin/env python3
"""Measure GigaAM voice ASR performance on the actual runner (issue #76).

Uses the exact pinned stack from ``aa.telegram.voice``:

- fussraider/GigaAM-Multilingual-sherpa-onnx-ctc @ pinned revision;
- large/model.int8.onnx (600M CTC INT8) + large/tokens.txt;
- sherpa-onnx==1.13.8, mono PCM float32 16 kHz, feature_dim 64, CPU only.

Generates a synthetic 10 s mono 16 kHz waveform, round-trips it through
OGG/Opus (ffmpeg) and the production ``FfmpegDecoder`` + ``GigaAMRecognizer``,
then records cold-load time, warm ASR latency, real-time factor and peak
RSS into ``qualification/voice_asr_perf.v1.json``.

No user data, no network service besides the pinned Hugging Face file
download. Never prints transcripts.
"""

from __future__ import annotations

import array
import datetime
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _synth_waveform(seconds: float = 10.0, sample_rate: int = 16000) -> list[float]:
    total = int(seconds * sample_rate)
    out: list[float] = []
    for n in range(total):
        t = n / sample_rate
        # Speech-band mixture: deterministic, no user data.
        sample = 0.3 * math.sin(2.0 * math.pi * 220.0 * t)
        sample += 0.2 * math.sin(2.0 * math.pi * 440.0 * t)
        sample += 0.1 * math.sin(2.0 * math.pi * 880.0 * t)
        out.append(sample)
    return out


def _peak_rss_mb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from aa.telegram.voice import (  # noqa: PLC0415
        FEATURE_DIM,
        MODEL_FILE,
        MODEL_REPO,
        MODEL_REVISION,
        NUM_THREADS,
        SAMPLE_RATE,
        SHERPA_ONNX_VERSION,
        TOKENS_FILE,
        FfmpegDecoder,
        GigaAMRecognizer,
        ensure_model_files,
    )

    model_dir = Path(os.environ.get("AA_VOICE_MODEL_DIR", str(root / "models" / "gigaam")))
    out_path = root / "qualification" / "voice_asr_perf.v1.json"

    ffmpeg_version = ""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15, check=False
        )
        ffmpeg_version = (proc.stdout.splitlines() or [""])[0][:200]
    except (FileNotFoundError, subprocess.SubprocessError):
        ffmpeg_version = "ffmpeg-missing"

    cold_start = time.perf_counter()
    import sherpa_onnx  # type: ignore[import-untyped]  # noqa: PLC0415

    installed = str(getattr(sherpa_onnx, "__version__", ""))
    recognizer = GigaAMRecognizer(model_dir, num_threads=NUM_THREADS)
    try:
        recognizer.ensure_loaded()
    except Exception as exc:
        print(f"voice perf: recognizer unavailable: {exc}", file=sys.stderr)
        return 2
    cold_load_seconds = time.perf_counter() - cold_start

    model_path = model_dir / "model.int8.onnx"
    tokens_path = model_dir / "tokens.txt"
    try:
        model_path, tokens_path = ensure_model_files(model_dir)
    except Exception as exc:
        print(f"voice perf: model provisioning failed: {exc}", file=sys.stderr)
        return 2

    duration_seconds = 10.0
    waveform = _synth_waveform(seconds=duration_seconds, sample_rate=SAMPLE_RATE)

    # OGG round-trip through ffmpeg so the measured path matches production
    # (OGG bytes -> FfmpegDecoder -> PCM -> ASR).
    decoder = FfmpegDecoder()
    with tempfile.TemporaryDirectory(prefix="aa-voice-perf-") as tmp:
        workdir = Path(tmp)
        pcm_path = workdir / "synth.pcm"
        ogg_path = workdir / "synth.ogg"
        raw = array.array("f", waveform).tobytes()
        pcm_path.write_bytes(raw)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "f32le",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-i",
                    str(pcm_path),
                    "-c:a",
                    "libopus",
                    str(ogg_path),
                ],
                timeout=60,
                check=True,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"voice perf: synth OGG encode failed: {exc}", file=sys.stderr)
            return 2
        ogg_bytes = ogg_path.read_bytes()
        decoded = decoder.decode(ogg_bytes, workdir=workdir)
        audio_seconds = len(decoded) / float(SAMPLE_RATE)

        # Warm-up then timed runs (transcript content never printed).
        recognizer.transcribe(decoded[:SAMPLE_RATE])
        latencies: list[float] = []
        for _ in range(3):
            begin = time.perf_counter()
            recognizer.transcribe(decoded)
            latencies.append(time.perf_counter() - begin)
        warm_latency = float(statistics.median(latencies))

    peak_rss = _peak_rss_mb()
    rtf = warm_latency / audio_seconds if audio_seconds > 0 else 0.0
    payload = {
        "schema_version": "voice-asr-perf-v1",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_file": MODEL_FILE,
        "tokens_file": TOKENS_FILE,
        "sherpa_onnx_version": SHERPA_ONNX_VERSION,
        "installed_sherpa_onnx_version": installed,
        "sample_rate_hz": SAMPLE_RATE,
        "feature_dim": FEATURE_DIM,
        "num_threads": NUM_THREADS,
        "cpu_inference_only": True,
        "model_bytes": model_path.stat().st_size,
        "tokens_bytes": tokens_path.stat().st_size,
        "audio_duration_seconds": audio_seconds,
        "cold_load_seconds": cold_load_seconds,
        "warm_asr_latency_seconds": warm_latency,
        "warm_latencies_seconds": latencies,
        "real_time_factor": rtf,
        "peak_rss_mb": peak_rss,
        "ffmpeg_version": ffmpeg_version,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "measured_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "runner": "ubuntu-latest-equivalent",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"cold_load_seconds={cold_load_seconds:.2f}")
    print(f"warm_latency_seconds={warm_latency:.2f}")
    print(f"rtf={rtf:.3f}")
    print(f"peak_rss_mb={peak_rss:.1f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
