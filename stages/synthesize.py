import logging
import time
from pathlib import Path

from config import Config
from models.tts_edge import EdgeTTSWrapper
from utils import StageResult, read_srt, get_audio_duration

logger = logging.getLogger("dubbing")

def synthesize(project_dir: Path, config: Config, progress=None) -> StageResult:
    try:
        pt_srt_path = project_dir / "work" / "pt.srt"
        if not pt_srt_path.exists():
            return StageResult(success=False, error="pt.srt not found. Run translate first.")
        work_dir = project_dir / "work"
        segments_dir = work_dir / "pt_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        cues = read_srt(pt_srt_path)
        tts_engine = config.tts_engine
        logger.info(f"[SYNTHESIZE] Engine: {tts_engine}, Cues: {len(cues)}")
        if tts_engine == "qwen":
            return _synthesize_qwen(project_dir, config, cues, segments_dir, progress)
        else:
            return _synthesize_edge(config, cues, segments_dir, progress)
    except Exception as e:
        logger.error(f"[SYNTHESIZE] Error: {e}")
        return StageResult(success=False, error=str(e))


def _qwen_reference(project_dir, config, progress):
    """Returns (audio_path, transcript_or_None, needs_br_accent) to clone the voice from.
    needs_br_accent is True when the reference speaks another language (the
    course's own narrator), so Qwen borrows its pt-BR accent from an example."""
    from models.audio_enhance import ReferenceAudioEnhancer, _estimate_snr_db, MIN_SNR_DB_TO_SKIP_ENHANCE
    from models.tts_qwen import pick_reference_window, cut_reference_clip, reference_scorer

    work_dir = project_dir / "work"
    if config.reference_audio_path and Path(config.reference_audio_path).exists():
        # A saved voice: cloned directly, with its transcript when we have one.
        lang = (config.reference_language or "").lower()
        foreign = lang not in ("pt", "portuguese")
        return config.reference_audio_path, (config.reference_text or None), foreign and config.QWEN_BR_ACCENT_EXAMPLE

    vocals_path = work_dir / "vocals.wav"
    if not vocals_path.exists():
        vocals_path = work_dir / "audio_full.wav"
    if not vocals_path.exists():
        return None, None, False

    window = None
    en_srt = work_dir / "en.srt"
    if en_srt.exists():
        window = pick_reference_window(read_srt(en_srt), reference_scorer(work_dir))
    if window is None:
        # No clean stretch of speech with known text: clone timbre only.
        from models.audio_enhance import enhance_reference_audio
        return str(enhance_reference_audio(vocals_path, work_dir / "vocals_enhanced.wav")), None, config.QWEN_BR_ACCENT_EXAMPLE

    start, end, text = window
    ref_path = cut_reference_clip(vocals_path, start, end, work_dir / "qwen_ref.wav")
    logger.info(f"[SYNTHESIZE] Qwen reference {start:.1f}-{end:.1f}s: {text[:80]}")
    if config.ENHANCE_REFERENCE_AUDIO and _estimate_snr_db(ref_path) < MIN_SNR_DB_TO_SKIP_ENHANCE:
        if progress:
            progress(0.03, desc="Enhancing reference audio...")
        try:
            enhancer = ReferenceAudioEnhancer()
            enhanced = work_dir / "qwen_ref_enhanced.wav"
            enhancer.enhance(ref_path, enhanced)
            enhancer.unload()
            ref_path = enhanced
        except Exception as e:
            logger.warning(f"[SYNTHESIZE] Reference enhance failed ({e}); using raw clip")
    if not config.QWEN_BR_ACCENT_EXAMPLE:
        # Timbre only: one consistent voice, the narrator's, for every line.
        return str(ref_path), None, False
    return str(ref_path), text, True


def _synthesize_qwen(project_dir, config, cues, segments_dir, progress):
    from models.tts_qwen import QwenTTSWrapper

    reference_audio, ref_text, br_accent = _qwen_reference(project_dir, config, progress)
    if not reference_audio:
        return StageResult(success=False, error="No reference audio for Qwen3-TTS voice cloning.")

    if progress:
        progress(0.05, desc="Loading Qwen3-TTS...")
    t_load_start = time.time()
    wrapper = QwenTTSWrapper(config)
    wrapper.load(reference_audio=reference_audio, ref_text=ref_text, br_accent=br_accent)
    t_load = time.time() - t_load_start
    logger.info(f"[SYNTHESIZE] Qwen3-TTS loaded in {t_load:.1f}s")
    # Qwen pronto (sem captura de CUDA graph rolando): o worker pode começar a
    # adiantar o próximo vídeo da fila em paralelo (PIPELINE_OVERLAP).
    hook = getattr(config, "_after_tts_load", None)
    if hook:
        try:
            hook()
        except Exception as e:
            logger.warning(f"[SYNTHESIZE] Overlap hook failed ({e})")
    total = len(cues)
    t_start = time.time()

    def _cue_progress(frac, desc):
        if progress:
            progress(0.10 + frac * 0.85, desc=desc)

    try:
        output_paths = wrapper.synthesize_batch(cues, segments_dir, progress=_cue_progress)
    except Exception as e:
        wrapper.unload()
        return StageResult(success=False, error=str(e))
    wrapper.unload()
    t_total = time.time() - t_start
    avg = t_total / total if total else 0
    logger.info(f"[SYNTHESIZE] Qwen done: {total} cues em {t_total:.1f}s ({avg:.2f}s/cue)")
    if progress:
        progress(1.0, desc=f"TTS done: {total} cues em {t_total:.1f}s ({avg:.2f}s/cue)")
    return StageResult(
        success=True,
        output_paths=[Path(p) for p in output_paths if p],
        metadata={"warnings": [], "time_s": round(t_total, 1), "avg_per_cue_s": round(avg, 3), "model_load_s": round(t_load, 1)},
    )


def _estimate_edge_rate(text: str, target_duration: float) -> str:
    char_count = len(text.replace(" ", ""))
    if char_count == 0 or target_duration <= 0:
        return "+0%"
    natural_duration = char_count / 7.0
    ratio = natural_duration / target_duration
    rate_pct = round((ratio - 1.0) * 100)
    rate_pct = max(-50, min(100, rate_pct))
    return f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"


def _synthesize_edge(config, cues, segments_dir, progress):
    total = len(cues)
    if progress:
        progress(0.05, desc=f"Synthesizing {total} cues with Edge-TTS...")
    wrapper = EdgeTTSWrapper(config)
    output_paths = []
    warnings = []
    t_start = time.time()
    for i, cue in enumerate(cues):
        out_path = str(segments_dir / f"{cue.index:03d}.wav")
        try:
            rate = _estimate_edge_rate(cue.text, cue.duration())
            wrapper.set_rate(rate)
            wrapper.synthesize(cue.text, out_path)
            dur = get_audio_duration(Path(out_path))
            if dur < 0.1:
                warnings.append(f"Cue {cue.index}: generated audio is silent")
        except Exception as e:
            warnings.append(f"Cue {cue.index}: synthesis failed - {e}")
        output_paths.append(Path(out_path))
        done = i + 1
        elapsed = time.time() - t_start
        avg = elapsed / done
        if progress:
            progress(0.05 + 0.9*(done/total), desc=f"Edge {done}/{total} | ETA {avg*(total-done):.0f}s")
    t_total = time.time() - t_start
    avg = t_total / total if total else 0
    logger.info(f"[SYNTHESIZE] Edge done: {total} cues em {t_total:.1f}s ({avg:.2f}s/cue)")
    if progress:
        progress(1.0, desc=f"TTS done: {total} cues em {t_total:.1f}s")
    return StageResult(success=True, output_paths=output_paths,
        metadata={"warnings": warnings, "time_s": round(t_total,1), "avg_per_cue_s": round(avg,3)})
