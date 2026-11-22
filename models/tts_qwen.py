import logging
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import Config
from models import model_cache
from utils import SRTCue

logger = logging.getLogger("dubbing")

MAX_TTS_CHARS = 250

# Translation output mangles decimals/versions ("Gamma 2 .2", "Rec .709"),
# which TTS models can't read and tend to babble on. Well-formed numbers get
# spelled out ("2,2" -> "dois vírgula dois").
_SPLIT_DECIMAL_RE = re.compile(r"(\d)\s+([.,])\s*(\d)")  # "2 .2" -> "2.2"
_LEADING_DOT_NUM_RE = re.compile(r"(?<![\w.,])\.(\d)")   # ".709" -> "709"
_DOT_DECIMAL_RE = re.compile(r"(\d)\.(\d{1,2}|\d{4,})\b")  # "2.2" -> "2,2" (pt decimal comma; "1.000" stays a thousand)
_REPEATED_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)  # "nós nós" -> "nós"

# Generous ceiling on how long a cue's audio may be: Portuguese speech runs at
# ~12-16 chars/s, so anything past ~6.7 chars/s plus a margin is babble,
# moaning or a hang, not speech.
_MAX_SEC_BASE = 1.5
_MAX_SEC_PER_CHAR = 0.15
_DIGIT_WEIGHT = 4  # a digit expands into a whole spoken word


def _clean_text(text: str) -> str:
    text = text.strip()
    words = text.split()
    if len(words) > 5:
        unique = set(words)
        if len(unique) < len(words) * 0.3:
            text = " ".join(list(dict.fromkeys(words))[:20])
    text = _SPLIT_DECIMAL_RE.sub(r"\1\2\3", text)
    text = _LEADING_DOT_NUM_RE.sub(r"\1", text)
    text = _DOT_DECIMAL_RE.sub(r"\1,\2", text)
    text = _REPEATED_WORD_RE.sub(r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:MAX_TTS_CHARS]


def _max_expected_seconds(text: str) -> float:
    chars = sum(_DIGIT_WEIGHT if c.isdigit() else 1 for c in text if not c.isspace())
    return _MAX_SEC_BASE + chars * _MAX_SEC_PER_CHAR


def _trim_silence(wav: np.ndarray, sr: int, pad_ms: float = 80.0) -> np.ndarray:
    """Cut leading/trailing silence and faint tail noise."""
    frame = int(sr * 0.02)
    n = len(wav) // frame
    if n < 3:
        return wav
    rms = np.sqrt((wav[: n * frame].reshape(n, frame).astype(np.float64) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    active = np.where(db > db.max() - 40)[0]
    if not len(active):
        return wav
    pad = int(sr * pad_ms / 1000)
    start = max(active[0] * frame - pad, 0)
    end = min((active[-1] + 1) * frame + pad, len(wav))
    return wav[start:end]

# Qwen's ICL cloning wants a few seconds of clean speech plus its exact transcript.
REF_MIN_SEC = 6.0
REF_MAX_SEC = 12.0
REF_MAX_GAP_SEC = 0.4
_MIN_SEC_FOR_TEXT = 0.3  # non-empty text that comes out shorter than this was dropped
_CODEC_FPS = 24000 / 1920  # Qwen3-TTS 12Hz tokenizer: 12.5 codec frames per second of audio


def pick_reference_window(en_cues: list[SRTCue], score_fn=None) -> tuple[float, float, str] | None:
    """Run of back-to-back English cues lasting REF_MIN_SEC..REF_MAX_SEC to clone from.
    Only looks at the middle 80% of the video (intros/outros carry music, other
    voices). With score_fn(start, end) -> float the cleanest window wins,
    otherwise the longest."""
    windows = []
    n = len(en_cues)
    first, last = (n // 10, n - n // 10) if n > 20 else (0, n)
    for i in range(first, last):
        start = en_cues[i].start
        j = i
        while (j + 1 < last
               and en_cues[j + 1].start - en_cues[j].end <= REF_MAX_GAP_SEC
               and en_cues[j + 1].end - start <= REF_MAX_SEC):
            j += 1
        if en_cues[j].end - start >= REF_MIN_SEC:
            windows.append((start, en_cues[j].end, " ".join(c.text for c in en_cues[i:j + 1])))
    if not windows:
        return None
    if score_fn is None:
        return max(windows, key=lambda w: w[1] - w[0])
    return max(windows, key=lambda w: score_fn(w[0], w[1]))


def reference_scorer(work_dir: Path):
    """Score a candidate clip: confident words, little music under the voice,
    longer is slightly better. Uses what earlier stages already produced."""
    import json
    words = []
    wp = work_dir / "en_words.json"
    if wp.exists():
        try:
            words = json.loads(wp.read_text(encoding="utf-8"))
        except Exception:
            words = []
    vocals, background = work_dir / "vocals.wav", work_dir / "background.wav"
    have_stems = vocals.exists() and background.exists()

    def rms(path, start, end):
        info = sf.info(str(path))
        data, _ = sf.read(str(path), start=int(start * info.samplerate), stop=int(end * info.samplerate))
        return float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if len(data) else 0.0

    def score(start, end):
        probs = [w.get("probability", 0.5) for w in words if start <= w["start"] < end]
        conf = sum(probs) / len(probs) if probs else 0.5
        clean = 1.0
        if have_stems:
            v, b = rms(vocals, start, end), rms(background, start, end)
            clean = v / (v + b + 1e-9)  # 1.0 = voice only, 0.5 = as loud as the music
        return conf * clean * (0.85 + 0.15 * min((end - start) / REF_MAX_SEC, 1.0))

    return score


def cut_reference_clip(vocals_path: Path, start: float, end: float, out_path: Path) -> Path:
    info = sf.info(str(vocals_path))
    pad = 0.1
    a = max(int((start - pad) * info.samplerate), 0)
    b = min(int((end + pad) * info.samplerate), info.frames)
    data, sr = sf.read(str(vocals_path), start=a, stop=b)
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(str(out_path), data, sr)
    return out_path


class QwenTTSWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.prompt = None
        self.fast = None

    def load(self, reference_audio: str, ref_text: str | None = None, br_accent: bool = False) -> None:
        self.fast = None
        use_fast = self.config.QWEN_BACKEND == "faster"
        if use_fast:
            try:
                import faster_qwen3_tts  # noqa: F401
            except ImportError:
                logger.warning("[QWEN] faster-qwen3-tts not installed; using the plain transformers backend")
                use_fast = False
        key = ("qwen3-tts", self.config.QWEN_TTS_MODEL, "faster" if use_fast else "hf")
        cached = model_cache.get(key)
        if cached is not None:
            logger.info("[QWEN] Reusing cached model (no reload)")
        else:
            from transformers.utils import logging as hf_logging
            # qwen_tts logs "<x>_config is None" on every load and every batch, and
            # transformers repeats "Setting pad_token_id" per generate() — pure noise.
            logging.getLogger("qwen_tts").setLevel(logging.WARNING)
            hf_logging.set_verbosity_error()
            logger.info(f"[QWEN] Loading {self.config.QWEN_TTS_MODEL} ({'CUDA graphs' if use_fast else 'transformers'})...")
            if use_fast:
                # Same weights, decode loop replayed as CUDA graphs: ~2x the
                # throughput of batched transformers generation on a 3060, less VRAM.
                from faster_qwen3_tts import FasterQwen3TTS
                cached = FasterQwen3TTS.from_pretrained(self.config.QWEN_TTS_MODEL)
                cached.warmup()
            else:
                from qwen_tts import Qwen3TTSModel
                cached = Qwen3TTSModel.from_pretrained(
                    self.config.QWEN_TTS_MODEL,
                    device_map="cuda:0",
                    dtype=torch.bfloat16,
                    # flash_attention_2 has no Windows wheels; sdpa is PyTorch's built-in fast path.
                    attn_implementation="sdpa",
                )
            model_cache.put(key, cached)
        if use_fast:
            self.fast = cached
            self.model = cached.model  # the underlying Qwen3TTSModel, for building voice prompts
        else:
            self.model = cached

        if br_accent:
            self.prompt = self._br_accent_prompt(reference_audio)
            if self.prompt is not None:
                return
        # Without a transcript only the speaker embedding is used (x-vector mode):
        # same timbre, a bit less of the original speaking style.
        self.prompt = self.model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=ref_text,
            x_vector_only_mode=not ref_text,
        )
        logger.info(f"[QWEN] Voice prompt ready ({'ICL' if ref_text else 'x-vector only'})")

    def _br_accent_prompt(self, reference_audio: str):
        """Timbre from the reference voice, speaking style from a pt-BR example.
        Cloning an English speaker straight into Portuguese drifts towards a
        European accent; the ICL example is what sets the accent, the speaker
        embedding is what sets whose voice it sounds like."""
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        accent_dir = Path(self.config.QWEN_ACCENT_DIR)
        examples = sorted(accent_dir.glob("*.wav")) if accent_dir.exists() else []
        if not examples:
            logger.warning(f"[QWEN] No pt-BR accent examples in {accent_dir}; cloning without them")
            return None

        target = self.model.create_voice_clone_prompt(ref_audio=reference_audio, x_vector_only_mode=True)[0]
        best, best_sim = None, -2.0
        for wav in examples:
            key = ("qwen3-accent", str(wav))
            item = model_cache.get(key)
            if item is None:
                text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
                item = model_cache.put(key, self.model.create_voice_clone_prompt(ref_audio=str(wav), ref_text=text)[0])
            # Pick the example closest to the target voice (e.g. male vs female),
            # so its intonation doesn't fight the timbre.
            sim = torch.nn.functional.cosine_similarity(
                item.ref_spk_embedding.float().flatten(), target.ref_spk_embedding.float().flatten(), dim=0
            ).item()
            if sim > best_sim:
                best, best_sim = (wav, item), sim

        wav, item = best
        logger.info(f"[QWEN] pt-BR accent from '{wav.stem}' (similarity {best_sim:.2f})")
        hybrid = [VoiceClonePromptItem(
            ref_code=item.ref_code,
            ref_spk_embedding=target.ref_spk_embedding,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=item.ref_text,
        )]
        if self.config.QWEN_ANCHOR_TRIES <= 0:
            return hybrid
        return self._self_anchor(hybrid, target.ref_spk_embedding, item.ref_text)

    def _self_anchor(self, hybrid, target_emb, text: str):
        """Replace the pt-BR example with the target voice saying it.
        With the stock example as ICL context, many cues drift towards the
        example speaker's timbre and the dub audibly switches voices between
        lines. So: say the example text a few times with the hybrid prompt, keep
        the take closest to the target voice, and use *that* as the ICL context —
        the context is now the target speaker already speaking pt-BR."""
        import librosa
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        spk_sr = self.model.model.speaker_encoder_sample_rate
        target = target_emb.float().flatten()
        self.prompt = hybrid
        best = None
        for _ in range(self.config.QWEN_ANCHOR_TRIES):
            wavs, sr = self._generate([text])
            take = _trim_silence(np.asarray(wavs[0], dtype=np.float32), sr)
            if self._looks_broken(take, sr, text):
                continue
            y = librosa.resample(take, orig_sr=sr, target_sr=spk_sr) if sr != spk_sr else take
            emb = torch.as_tensor(self.model.model.extract_speaker_embedding(audio=y, sr=spk_sr)).float().flatten()
            sim = torch.nn.functional.cosine_similarity(emb.to(target.device), target, dim=0).item()
            if best is None or sim > best[0]:
                best = (sim, take, sr)
        if best is None:
            logger.warning("[QWEN] Self-anchor takes all broken; keeping the stock pt-BR example")
            return hybrid
        sim, take, sr = best
        item = self.model.create_voice_clone_prompt(ref_audio=(take, sr), ref_text=text)[0]
        logger.info(f"[QWEN] Voice prompt ready (self-anchored pt-BR take, similarity to target {sim:.3f})")
        return [VoiceClonePromptItem(
            ref_code=item.ref_code,
            ref_spk_embedding=target_emb,
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=text,
        )]

    def _generate(self, texts: list[str], **kwargs) -> tuple[list[np.ndarray], int]:
        # The checkpoint allows 8192 codec frames (~11 min) per item, and a batch
        # runs until its longest item stops — so one cue that never stops holds
        # the whole batch for minutes. Cap at a bit over the "broken" threshold:
        # a runaway gets cut early and still fails _looks_broken, so it's retried.
        if self.fast is not None:
            # CUDA-graph path decodes one item at a time (no batching).
            wavs, sr = [], 24000
            for t in texts:
                out, sr = self.fast.generate_voice_clone(
                    text=t,
                    language=self.config.QWEN_LANGUAGE,
                    voice_clone_prompt=self.prompt,
                    max_new_tokens=int(_CODEC_FPS * _max_expected_seconds(t) * 1.15) + 12,
                    temperature=kwargs.get("temperature", self.config.QWEN_TEMPERATURE),
                    append_silence=False,
                )
                wavs.append(np.asarray(out[0], dtype=np.float32).flatten())
            return wavs, sr
        longest = max(_max_expected_seconds(t) for t in texts)
        kwargs.setdefault("max_new_tokens", int(_CODEC_FPS * longest * 1.15) + 12)
        kwargs.setdefault("temperature", self.config.QWEN_TEMPERATURE)
        return self.model.generate_voice_clone(
            text=texts,
            language=[self.config.QWEN_LANGUAGE] * len(texts),
            voice_clone_prompt=self.prompt * len(texts),
            **kwargs,
        )

    def _looks_broken(self, wav: np.ndarray, sr: int, text: str) -> bool:
        dur = len(wav) / sr
        too_short = dur < _MIN_SEC_FOR_TEXT and len(text.strip()) >= 4  # "Ok." can be short
        return too_short or dur > _max_expected_seconds(text)

    def synthesize_batch(self, cues: list[SRTCue], segments_dir: Path, progress=None) -> list[str | None]:
        items = [(c, _clean_text(c.text)) for c in cues]
        results: dict[int, str | None] = {}
        batch_size = self.config.QWEN_BATCH_SIZE
        # Sorted by length so each batch holds similar-length cues: a batch takes
        # as long as its longest cue, so mixing one long line with short ones
        # leaves the GPU padding the short ones for nothing.
        todo = sorted(((c, t) for c, t in items if t), key=lambda ct: len(ct[1]))
        for c, t in items:
            if not t:
                results[c.index] = None

        # Batch by text length as well as count: VRAM grows with the longest
        # output in the batch, and 8 long cues at once is what filled a 12GB card.
        batches, cur, cur_chars = [], [], 0
        for item in todo:
            if cur and (len(cur) >= batch_size or cur_chars + len(item[1]) > self.config.QWEN_BATCH_CHARS):
                batches.append(cur)
                cur, cur_chars = [], 0
            cur.append(item)
            cur_chars += len(item[1])
        if cur:
            batches.append(cur)

        done = 0
        for batch in batches:
            wavs, sr = self._generate([t for _, t in batch])
            for (cue, text), wav in zip(batch, wavs):
                wav = np.asarray(wav, dtype=np.float32)
                attempt = 0
                while self._looks_broken(wav, sr, text) and attempt < self.config.QWEN_MAX_RETRIES:
                    attempt += 1
                    logger.warning(
                        f"[QWEN] Cue {cue.index}: {len(wav) / sr:.1f}s for {len(text)} chars, retry {attempt}"
                    )
                    retry, sr = self._generate([text], temperature=max(self.config.QWEN_TEMPERATURE - 0.2 * attempt, 0.3))
                    wav = np.asarray(retry[0], dtype=np.float32)
                wav = _trim_silence(wav, sr)
                out_path = segments_dir / f"{cue.index:03d}.wav"
                sf.write(str(out_path), wav, sr)
                results[cue.index] = str(out_path)
            done += len(batch)
            if progress:
                progress(done / max(len(todo), 1), f"{done}/{len(todo)} cues")

        return [results.get(c.index) for c in cues]

    def unload(self):
        # Model stays in model_cache for the next job; drop per-video voice state.
        self.prompt = None
        self.model = None
        self.fast = None
        torch.cuda.empty_cache()
