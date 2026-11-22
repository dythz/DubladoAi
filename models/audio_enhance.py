import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("dubbing")

# Voice cloning only needs a short, clean clip to compute good speaker latents — feeding
# it (and DeepFilterNet) an entire multi-minute vocals track wastes time and
# is what triggered a cuDNN GRU crash on long tensors. Trim to a short window first.
REFERENCE_CLIP_SECONDS = 45

# Skip DeepFilterNet entirely when the clip is already clean — running a denoiser
# on already-clean speech buys nothing and costs a few seconds per video for no reason.
MIN_SNR_DB_TO_SKIP_ENHANCE = 20.0


class ReferenceAudioEnhancer:
    """Denoises/enhances a speaker reference clip before voice cloning.

    Cloning a voice from a noisy/echoey/lo-fi clip is the main cause of TTS
    producing garbled or moaning output, since the model ends up cloning the
    noise along with the voice.

    DeepFilterNet auto-picks CUDA when available. That's fine and fast (~5s)
    once the input is trimmed to a short clip — the original crash
    (`CUDNN_STATUS_NOT_SUPPORTED`) came from feeding it a full 14-minute
    track, not from running on GPU. Kept on GPU for speed, with a CPU retry
    as a safety net in case a particular clip still trips a CUDA/cuDNN error.
    """

    def __init__(self):
        self.model = None
        self.df_state = None

    def load(self, force_cpu: bool = False) -> None:
        if force_cpu:
            import torch
            import df.enhance as _df_enhance
            _df_enhance.get_device = lambda: torch.device("cpu")
        from df.enhance import init_df
        self.model, self.df_state, _ = init_df()

    def enhance(self, input_path: Path, output_path: Path) -> Path:
        from df.enhance import enhance, load_audio, save_audio
        if self.model is None:
            self.load()
        try:
            audio, _ = load_audio(str(input_path), sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
        except RuntimeError as e:
            logger.warning(f"[ENHANCE] GPU run failed ({e}), retrying on CPU")
            self.unload()
            self.load(force_cpu=True)
            audio, _ = load_audio(str(input_path), sr=self.df_state.sr())
            enhanced = enhance(self.model, self.df_state, audio)
        save_audio(str(output_path), enhanced, self.df_state.sr())
        return output_path

    def unload(self) -> None:
        self.model = None
        self.df_state = None


def _trim_reference_clip(input_path: Path, output_path: Path, seconds: int = REFERENCE_CLIP_SECONDS) -> Path:
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path), "-t", str(seconds),
        "-ar", "44100", "-ac", "1", str(output_path),
    ], capture_output=True, check=True, timeout=60)
    return output_path


def _estimate_snr_db(path: Path, frame_ms: float = 20.0) -> float:
    """Cheap noise-floor-vs-peak proxy for "is this clip already clean".
    Not a true SNR measurement, just fast (<50ms) and good enough to decide
    whether running a denoiser is worth the time."""
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    frame_len = max(int(sr * frame_ms / 1000), 1)
    n_frames = len(data) // frame_len
    if n_frames < 5:
        return 0.0  # too short to judge — treat as noisy, enhance to be safe
    frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    noise_floor = max(np.percentile(rms, 10), 1e-6)
    signal_level = max(np.percentile(rms, 90), 1e-6)
    return 20.0 * np.log10(signal_level / noise_floor)


def enhance_reference_audio(input_path: Path, output_path: Path) -> Path:
    """One-shot helper: trim to a short clip, then denoise/enhance it for
    voice cloning — but only if it actually looks noisy. Falls back gracefully
    (trimmed-but-unenhanced, or the original file untouched) at each step — a
    missing/broken enhancer should never block the pipeline."""
    trimmed_path = output_path.with_name(output_path.stem + "_trimmed.wav")
    try:
        _trim_reference_clip(input_path, trimmed_path)
        reference_source = trimmed_path
    except Exception as e:
        logger.warning(f"[ENHANCE] Failed to trim reference clip ({e}); using full file")
        reference_source = input_path

    try:
        snr_db = _estimate_snr_db(reference_source)
        if snr_db >= MIN_SNR_DB_TO_SKIP_ENHANCE:
            logger.info(f"[ENHANCE] Reference clip already clean (~{snr_db:.1f}dB), skipping denoise")
            return reference_source
    except Exception as e:
        logger.warning(f"[ENHANCE] SNR check failed ({e}); enhancing anyway")

    try:
        enhancer = ReferenceAudioEnhancer()
        enhancer.enhance(reference_source, output_path)
        enhancer.unload()
        logger.info(f"[ENHANCE] Reference audio enhanced: {reference_source.name} -> {output_path.name}")
        return output_path
    except Exception as e:
        logger.warning(f"[ENHANCE] Failed to enhance reference audio ({e}); using {reference_source.name} as-is")
        return reference_source
