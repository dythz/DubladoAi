import logging
import time
import torchaudio
from pathlib import Path

from config import Config
from models.demucs_model import DemucsModelWrapper
from utils import StageResult

logger = logging.getLogger("dubbing")


def _background_level_db(whisper_wav: Path) -> float | None:
    """Volume (dBFS) of what plays in the pauses between sentences: the 90th
    percentile of 100 ms frames outside speech. A narration-only lesson sits
    around -60 dBFS there; background music is usually -40 or louder.
    None = couldn't judge (too little pause, or VAD unavailable)."""
    import numpy as np
    import soundfile as sf
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio, sr = sf.read(str(whisper_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000 or len(audio) < sr * 10:
        return None
    speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=300))
    mask = np.ones(len(audio), dtype=bool)  # True = pause
    pad = int(0.25 * sr)  # keep away from word edges/breaths
    for seg in speech:
        mask[max(0, seg["start"] - pad): seg["end"] + pad] = False
    frame = sr // 10
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    pause_frames = frames[mask[: n * frame].reshape(n, frame).all(axis=1)]
    if len(pause_frames) < 30:  # under 3 s of pauses in the whole video
        return None
    rms = np.sqrt((pause_frames ** 2).mean(axis=1)) + 1e-9
    return float(np.percentile(20 * np.log10(rms), 90))


def _write_without_separation(audio_path: Path, vocals_path: Path, background_path: Path) -> None:
    """No music: the full track is the voice, and the background is silence."""
    import numpy as np
    import soundfile as sf
    info = sf.info(str(audio_path))
    with sf.SoundFile(str(audio_path)) as src, \
            sf.SoundFile(str(vocals_path), "w", info.samplerate, info.channels, "PCM_16") as voc, \
            sf.SoundFile(str(background_path), "w", info.samplerate, info.channels, "PCM_16") as bg:
        for block in src.blocks(blocksize=info.samplerate * 30, dtype="float32", always_2d=True):
            voc.write(block)
            bg.write(np.zeros_like(block))


def separate_vocals(project_dir: Path, config: Config, progress=None) -> StageResult:
    """Stage 2: Separate vocals from background using Demucs htdemucs."""
    try:
        audio_path = project_dir / "work" / "audio_full.wav"
        if not audio_path.exists():
            return StageResult(success=False, error="audio_full.wav not found. Run extract first.")

        work_dir = project_dir / "work"
        vocals_path = work_dir / "vocals.wav"
        background_path = work_dir / "background.wav"
        logger.info(f"[SEPARATE] Input: {audio_path.name} | model: {config.DEMUCS_MODEL} | shifts: {config.DEMUCS_SHIFTS}")

        if getattr(config, "AUTO_SKIP_SEPARATION", False):
            whisper_wav = work_dir / "audio_whisper.wav"
            try:
                level = _background_level_db(whisper_wav) if whisper_wav.exists() else None
            except Exception as e:
                logger.warning(f"[SEPARATE] Music check failed ({e}); running Demucs")
                level = None
            if level is not None and level < config.SEPARATION_SILENCE_DB:
                t0 = time.time()
                _write_without_separation(audio_path, vocals_path, background_path)
                logger.info(
                    f"[SEPARATE] No background music (pauses at {level:.0f} dBFS < "
                    f"{config.SEPARATION_SILENCE_DB:.0f}) - skipped Demucs ({time.time() - t0:.1f}s)"
                )
                if progress:
                    progress(1.0, desc="Sem musica de fundo - separacao pulada")
                return StageResult(
                    success=True,
                    output_paths=[vocals_path, background_path],
                    metadata={"skipped": "no_music", "pause_level_db": round(level, 1)},
                )
            if level is not None:
                logger.info(f"[SEPARATE] Background at {level:.0f} dBFS in pauses - running Demucs")

        if progress:
            progress(0.1, desc="Loading Demucs model...")

        t0 = time.time()
        wrapper = DemucsModelWrapper(config)
        wrapper.load()
        logger.info(f"[SEPARATE] Model ready in {time.time() - t0:.1f}s")

        if progress:
            progress(0.3, desc="Separating vocals from background...")

        t0 = time.time()
        result = wrapper.separate(audio_path)
        logger.info(f"[SEPARATE] Separation done in {time.time() - t0:.1f}s")

        if progress:
            progress(0.8, desc="Saving separated tracks...")

        torchaudio.save(str(vocals_path), result["vocals"], result["sr"])
        torchaudio.save(str(background_path), result["background"], result["sr"])
        logger.info(
            f"[SEPARATE] Saved vocals ({vocals_path.stat().st_size / 1024 / 1024:.1f} MB) "
            f"and background ({background_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )

        wrapper.unload()

        if progress:
            progress(1.0, desc="Vocal separation complete")

        return StageResult(
            success=True,
            output_paths=[vocals_path, background_path],
        )
    except Exception as e:
        logger.error(f"[SEPARATE] Error: {e}")
        return StageResult(success=False, error=str(e))