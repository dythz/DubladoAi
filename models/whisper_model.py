import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from pathlib import Path
import logging
import re

from config import Config
from gpu_manager import GPUManager
from models import model_cache

logger = logging.getLogger("dubbing")

# Phrases Whisper invents over music/silence (it was trained on YouTube captions).
_HALLUCINATIONS = (
    "thank you for watching", "thanks for watching", "please subscribe", "subscribe to",
    "like and subscribe", "see you in the next video", "subtitles by", "captions by",
    "amara.org", "transcribed by",
)


def _non_speech_reason(seg) -> str | None:
    """Why a Whisper segment is probably music/noise rather than the narrator."""
    text = seg.text.strip()
    norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
    if not norm or "\u266a" in text or norm in ("music", "applause", "laughter"):
        return "music/no words"
    if any(h in norm for h in _HALLUCINATIONS):
        return "known hallucination"
    # Whisper's own "this isn't speech" signal, as used by its reference decoder.
    no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
    logprob = getattr(seg, "avg_logprob", 0.0) or 0.0
    if no_speech > 0.6 and logprob < -0.8:
        return f"no speech p={no_speech:.2f}"
    # Song lyrics / looping garbage compress far better than normal speech.
    ratio = getattr(seg, "compression_ratio", 0.0) or 0.0
    if ratio > 2.4:
        return f"repetitive ratio={ratio:.1f}"
    if logprob < -1.2:
        return f"low confidence {logprob:.2f}"
    return None


class WhisperModelWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load(self) -> None:
        GPUManager.acquire("transcribe")
        key = ("whisper", self.config.WHISPER_MODEL, self.config.WHISPER_DEVICE, self.config.WHISPER_COMPUTE_TYPE)
        cached = model_cache.get(key)
        if cached is not None:
            self.model = cached
        else:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.config.WHISPER_MODEL,
                device=self.config.WHISPER_DEVICE,
                compute_type=self.config.WHISPER_COMPUTE_TYPE,
                download_root=str(self.config.MODEL_CACHE_DIR / "whisper"),
                num_workers=4,
            )
            model_cache.put(key, self.model)
        GPUManager.set_model(self.model)

    def _transcribe_batched(self, audio_path: Path):
        """faster-whisper's batched mode: the VAD cuts the audio into chunks and
        several chunks decode at once. Big win on large GPUs; returns None if
        this faster-whisper has no batched mode so the caller falls back."""
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError:
            logger.info("[TRANSCRIBE] Batched mode needs faster-whisper>=1.1; using sequential")
            return None
        pipe = BatchedInferencePipeline(model=self.model)
        segments, _info = pipe.transcribe(
            str(audio_path),
            language=self.config.WHISPER_LANGUAGE,
            word_timestamps=True,
            beam_size=self.config.WHISPER_BEAM_SIZE,
            batch_size=self.config.WHISPER_BATCH_SIZE,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return list(segments)

    def transcribe(self, audio_path: Path) -> list[dict]:
        segments = None
        if self.config.WHISPER_BATCH_SIZE > 1:
            try:
                segments = self._transcribe_batched(audio_path)
                if segments is not None:
                    logger.info(f"[TRANSCRIBE] Batched mode (batch {self.config.WHISPER_BATCH_SIZE})")
            except Exception as e:
                logger.warning(f"[TRANSCRIBE] Batched mode failed ({e}); using sequential")
                segments = None
        if segments is None:
            segments = self._transcribe_sequential(audio_path)
        return self._collect_words(segments)

    def _transcribe_sequential(self, audio_path: Path):
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.WHISPER_LANGUAGE,
            word_timestamps=True,
            vad_filter=self.config.WHISPER_VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=500),
            beam_size=self.config.WHISPER_BEAM_SIZE,
            best_of=self.config.WHISPER_BEAM_SIZE,
            # Conditioning on previous text is what lets one hallucination over an
            # intro song snowball into the next segments.
            condition_on_previous_text=False,
        )
        return segments

    def _collect_words(self, segments) -> list[dict]:
        words = []
        for seg in segments:
            reason = _non_speech_reason(seg)
            if reason:
                logger.warning(f"[TRANSCRIBE] Dropped {seg.start:.1f}-{seg.end:.1f}s ({reason}): {seg.text.strip()[:70]!r}")
                continue
            for w in (seg.words or []):
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                })
        return words

    def unload(self) -> None:
        # Cached in model_cache — stays loaded for the next job.
        self.model = None
        GPUManager.release("transcribe")
