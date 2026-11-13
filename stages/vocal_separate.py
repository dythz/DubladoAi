import logging
import time
import torchaudio
from pathlib import Path

from config import Config
from models.demucs_model import DemucsModelWrapper
from utils import StageResult

logger = logging.getLogger("dubbing")


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