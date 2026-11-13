import logging
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import Config
from utils import (
    StageResult, SRTCue, read_srt, get_audio_duration,
    stretch_audio, assemble_dubbed_vocals,
)

logger = logging.getLogger("dubbing")


def time_sync(project_dir: Path, config: Config, progress=None) -> StageResult:
    """Stage 6: Time-stretch/pad synthesized audio to match SRT timing.
    Prefers speed adjustment and padding over cutting audio."""
    try:
        pt_srt_path = project_dir / "work" / "pt.srt"
        if not pt_srt_path.exists():
            return StageResult(success=False, error="pt.srt not found. Run translate first.")

        work_dir = project_dir / "work"
        segments_dir = work_dir / "pt_segments"
        synced_dir = work_dir / "pt_synced"
        synced_dir.mkdir(parents=True, exist_ok=True)

        cues = read_srt(pt_srt_path)
        if not cues:
            return StageResult(success=False, error="No cues found in pt.srt")

        total = len(cues)
        logger.info(f"[SYNC] Syncing {total} cues")

        if progress:
            progress(0.05, desc=f"Syncing {total} cues to SRT timing...")

        warnings = []
        method_counts = {}
        t_start = time.time()

        audio_full = work_dir / "audio_full.wav"
        track_end = get_audio_duration(audio_full) if audio_full.exists() else 0.0
        if track_end <= 0:
            track_end = cues[-1].end + 1.0

        # Elastic timeline: each cue starts at its SRT time or right after the
        # previous dubbed cue finishes, whichever is later — audio is never cut.
        starts: dict[int, float] = {}
        prev_end = 0.0
        drifts = []

        for i, cue in enumerate(cues):
            synth_path = segments_dir / f"{cue.index:03d}.wav"
            synced_path = synced_dir / f"{cue.index:03d}.wav"
            start = max(cue.start, prev_end + config.SYNC_MIN_GAP_SEC)
            drift = start - cue.start
            drifts.append(drift)
            next_start = cues[i + 1].start if i + 1 < len(cues) else track_end

            if not synth_path.exists():
                warnings.append(f"Cue {cue.index}: segment file not found")
                method = "missing"
                start = cue.start
                synced_path.unlink(missing_ok=True)  # don't let a stale file from an earlier run play
            else:
                t_cue = time.time()
                try:
                    result = _sync_cue_audio(cue, synth_path, synced_path, config, start, next_start, drift)
                except Exception as e:
                    logger.warning(f"[SYNC] Cue {cue.index} failed ({e}), leaving it silent")
                    synced_path.unlink(missing_ok=True)
                    result = {"ratio": 0, "speed": 0, "method": "error_silence", "duration": 0.0,
                              "warnings": [f"Cue {cue.index} error: {e}"]}
                cue_time = time.time() - t_cue
                method = result["method"]
                if result.get("warnings"):
                    warnings.extend(result["warnings"])
                prev_end = start + result["duration"]

                logger.info(
                    f"[SYNC] {i + 1}/{total} ({(i + 1) / total * 100:.0f}%) | "
                    f"cue {cue.index} | SRT {cue.duration():.2f}s | ratio {result['ratio']:.2f} | "
                    f"speed {result['speed']:.2f} | drift {drift:+.2f}s | method: {method} | {cue_time:.2f}s"
                )

            starts[cue.index] = start
            method_counts[method] = method_counts.get(method, 0) + 1

            done = i + 1
            elapsed = time.time() - t_start
            avg = elapsed / done
            remaining = avg * (total - done)

            if progress:
                frac = 0.05 + 0.6 * (done / total)
                progress(frac, desc=f"Sync {done}/{total} ({done / total * 100:.0f}%) | {elapsed:.1f}s | ETA {remaining:.0f}s")

        if progress:
            progress(0.70, desc="Assembling dubbed vocals track...")

        t_assemble = time.time()
        total_duration = track_end

        fade_ms = getattr(config, 'FADE_MS', 30.0)
        crossfade_ms = getattr(config, 'CROSSFADE_MS', 50.0)
        dubbed_vocals_path = assemble_dubbed_vocals(
            synced_dir, cues, total_duration,
            fade_ms=fade_ms, crossfade_ms=crossfade_ms, starts=starts,
        )
        assemble_time = time.time() - t_assemble
        drift_arr = np.array(drifts) if drifts else np.zeros(1)
        drift_stats = {
            "mean_s": round(float(drift_arr.mean()), 2),
            "p95_s": round(float(np.percentile(drift_arr, 95)), 2),
            "max_s": round(float(drift_arr.max()), 2),
        }
        logger.info(f"[SYNC] Dub lag behind original: {drift_stats}")

        t_total = time.time() - t_start
        avg_per_cue = t_total / total if total else 0
        logger.info(
            f"[SYNC] Done: {total} cues synced in {t_total:.1f}s "
            f"({avg_per_cue:.2f}s/cue) | assembly {assemble_time:.1f}s | "
            f"methods: {method_counts} | output: {dubbed_vocals_path} ({total_duration:.1f}s)"
        )

        if progress:
            progress(1.0, desc=f"Sync done: {total} cues in {t_total:.1f}s")

        return StageResult(
            success=True,
            output_paths=[synced_dir, dubbed_vocals_path],
            metadata={
                "warnings": warnings,
                "cue_count": total,
                "time_s": round(t_total, 1),
                "avg_per_cue_s": round(avg_per_cue, 3),
                "assemble_s": round(assemble_time, 1),
                "method_counts": method_counts,
                "drift": drift_stats,
            },
        )
    except Exception as e:
        return StageResult(success=False, error=str(e))


def _sync_cue_audio(
    cue: SRTCue, synth_path: Path, output_path: Path, config: Config,
    start: float, next_start: float, drift: float,
) -> dict:
    """Pick a playback speed for one cue placed at `start` (>= its SRT start).
    Never cuts or fades the speech: it may run past its own slot into the gap
    before the next cue and, if still too long, past `next_start` — the
    caller then pushes the next cue later. Returns the resulting duration."""
    synth_duration = get_audio_duration(synth_path)

    if synth_duration <= 0:
        output_path.unlink(missing_ok=True)
        return {"ratio": 0, "speed": 0, "method": "silence", "duration": 0.0,
                "warnings": [f"Cue {cue.index}: empty audio, left silent"]}

    tolerance = config.TOLERANCE_RATIO
    max_speed = config.MAX_SPEED_RATIO
    if drift > config.SYNC_CATCHUP_AFTER_SEC:
        max_speed = max(max_speed, config.SYNC_CATCHUP_SPEED)
    min_speed = config.MIN_SPEED_RATIO

    slot = max(cue.end - start, 0.2)               # what's left of its own subtitle slot
    window = max(next_start - start, slot)         # ...plus the pause before the next cue
    ratio = synth_duration / max(cue.duration(), 0.01)
    warnings = []

    if synth_duration > window * (1 + tolerance):
        speed = min(synth_duration / window, max_speed)
        method = "compress" if speed >= synth_duration / window else "compress+push"
    elif synth_duration < slot * (1 - tolerance):
        speed = max(synth_duration / slot, min_speed)
        method = "stretch" if speed < 1 else "none"
    else:
        speed = 1.0
        method = "none"

    if abs(speed - 1.0) < 0.01:
        shutil.copy2(synth_path, output_path)
        duration = synth_duration
    else:
        stretch_audio(synth_path, output_path, speed, method=config.STRETCH_METHOD)
        duration = get_audio_duration(output_path) or synth_duration / speed

    allowed = next_start + config.SYNC_MAX_DRIFT_SEC - start
    if duration > allowed:
        _trim_with_fade(output_path, max(allowed, 0.3))
        warnings.append(
            f"Cue {cue.index}: {duration:.1f}s still far too long at {speed:.2f}x, trimmed to "
            f"{max(allowed, 0.3):.1f}s to keep the dub within {config.SYNC_MAX_DRIFT_SEC:.0f}s of the video"
        )
        return {"ratio": ratio, "speed": speed, "method": "compress+trim",
                "duration": max(allowed, 0.3), "warnings": warnings}

    if method == "compress+push":
        warnings.append(
            f"Cue {cue.index}: {speed:.2f}x, runs {start + duration - next_start:.1f}s into next cue (pushed later)"
        )
    return {"ratio": ratio, "speed": speed, "method": method, "duration": duration, "warnings": warnings}


def _trim_with_fade(path: Path, keep_sec: float, fade_sec: float = 0.15) -> None:
    data, sr = sf.read(str(path))
    keep = min(int(keep_sec * sr), len(data))
    data = data[:keep]
    n = min(int(fade_sec * sr), len(data))
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n)
        data[-n:] = data[-n:] * (ramp[:, None] if data.ndim > 1 else ramp)
    sf.write(str(path), data, sr)
