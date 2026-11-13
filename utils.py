import logging
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("dubbing")


@dataclass
class SRTCue:
    index: int
    start: float
    end: float
    text: str

    def to_srt_block(self) -> str:
        return f"{self.index}\n{_format_srt_time(self.start)} --> {_format_srt_time(self.end)}\n{self.text}\n"

    def duration(self) -> float:
        return self.end - self.start


@dataclass
class StageResult:
    success: bool
    output_paths: list = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    total_ms = max(total_ms, 0)
    hrs, rem = divmod(total_ms, 3600000)
    mins, rem = divmod(rem, 60000)
    secs_val, ms = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs_val:02d},{ms:03d}"


def _parse_srt_time(t: str) -> float:
    t = t.strip().replace(",", ".")
    parts = t.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def write_srt(cues: list[SRTCue], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for cue in cues:
            f.write(cue.to_srt_block())
            f.write("\n")


def read_srt(path: Path) -> list[SRTCue]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})\s+(.*?)(?=\n\d+\s|\Z)",
        re.DOTALL,
    )
    cues = []
    for m in pattern.finditer(content):
        cues.append(SRTCue(
            index=int(m.group(1)),
            start=_parse_srt_time(m.group(2)),
            end=_parse_srt_time(m.group(3)),
            text=m.group(4).strip().replace("\n", " "),
        ))
    return cues


def dedupe_repeated_words(words: list[dict], max_repeat: int = 2) -> list[dict]:
    """Drop runs of the same word repeated beyond max_repeat.
    Whisper occasionally hallucinates a single word dozens of times in a row
    (silence/music/artifact segments — commonly "Meme", "you", "thank you"...).
    Left alone that garbage flows straight into the SRT, the translation, and
    then gets read out loud by the TTS. Collapsing it here, before cues are
    even built, keeps it out of every downstream stage at once."""
    import re as _re
    kept: list[dict] = []
    run_word: str | None = None
    run_count = 0
    dropped = 0
    for w in words:
        normalized = _re.sub(r"[^\w]", "", w["word"]).lower()
        if normalized and normalized == run_word:
            run_count += 1
        else:
            run_word = normalized
            run_count = 1
        if not normalized or run_count <= max_repeat:
            kept.append(w)
        else:
            dropped += 1
    if dropped:
        logger.warning(f"[TRANSCRIBE] Dropped {dropped} repeated-word hallucination tokens")
    return kept


def group_words_to_cues(
    words: list[dict],
    words_per_cue: int = 7,
    max_words_per_cue: int = 10,
    max_chars_per_cue: int = 45,
    min_duration: float = 1.0,
    min_gap: float = 0.08,
    max_duration: float = 10.0,
) -> list[SRTCue]:
    """Group word-level timestamps into SRT cues.
    Breaks at sentence/clause ends, char limits, word limits, or duration limits."""
    effective_max = max(max_words_per_cue, words_per_cue)
    cues = []
    current = []

    sentence_end = {".", "!", "?"}
    clause_end = {",", ";", ":", "-"}

    def current_text():
        return " ".join(w["word"].strip() for w in current)

    def flush():
        if not current:
            return
        start = current[0]["start"]
        end = current[-1]["end"]
        text = current_text()
        # Remove trailing periods from subtitle text
        text = text.rstrip(".")
        if end - start < min_duration:
            end = start + min_duration
        cues.append(SRTCue(index=len(cues) + 1, start=start, end=end, text=text))
        current.clear()

    for w in words:
        word_text = w["word"].strip()
        # Check if adding this word would exceed char limit
        test_text = (current_text() + " " + word_text).strip() if current else word_text
        would_exceed_chars = len(test_text) > max_chars_per_cue and len(current) > 0

        current.append(w)

        last_char = word_text[-1] if word_text else ""

        should_break = False
        if last_char in sentence_end:
            should_break = True
        elif len(current) >= words_per_cue and last_char in clause_end:
            should_break = True
        elif len(current) >= effective_max:
            should_break = True
        elif current[-1]["end"] - current[0]["start"] >= max_duration:
            should_break = True
        elif would_exceed_chars:
            should_break = True

        if should_break:
            flush()

    if current:
        flush()

    # Fix overlaps and ensure gaps
    for i in range(1, len(cues)):
        if cues[i].start < cues[i - 1].end + min_gap:
            cues[i].start = cues[i - 1].end + min_gap
        # Pushing start forward to fix an overlap can leave a cue technically valid
        # (start < end) but far too short (e.g. 0.08s) — that's not caught by an
        # end<=start check alone, and a slot that tiny forces the sync stage to fade
        # the dubbed audio out almost immediately, chopping off nearly the whole line.
        if cues[i].end - cues[i].start < min_duration:
            cues[i].end = cues[i].start + min_duration

    return cues


# --- FFmpeg helpers ---

def get_audio_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, errors='replace', timeout=15,
        )
        if result.returncode != 0:
            return 0.0
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def get_video_duration(path: Path) -> float:
    return get_audio_duration(path)


def stretch_audio(input_path: Path, output_path: Path, ratio: float, method: str = "atempo") -> None:
    """Time-stretch audio. ratio = original_duration / target_duration.
    ratio > 1 = speed up (compress), ratio < 1 = slow down (stretch).
    Always resample to 44100 Hz for consistency."""
    if method == "rubberband":
        # rubberband tempo is a speed multiplier: >1 = faster
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"rubberband=tempo={ratio:.6f}",
            "-ar", "44100", "-ac", "1", "-vn", str(output_path),
        ], capture_output=True, check=True, timeout=60)
    else:  # atempo
        atempo_chain = _compute_atempo_chain(ratio)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", atempo_chain,
            "-ar", "44100", "-ac", "1", "-vn", str(output_path),
        ], capture_output=True, check=True, timeout=60)


def _compute_atempo_chain(target_ratio: float) -> str:
    """Compute chained atempo filter values for ratios outside [0.5, 2.0]."""
    if target_ratio <= 0:
        return "atempo=1.0"
    if 0.5 <= target_ratio <= 2.0:
        return f"atempo={target_ratio:.6f}"
    n = 1
    while n < 20:  # safety limit
        each = target_ratio ** (1.0 / n)
        if 0.5 <= each <= 2.0:
            break
        n += 1
    return ",".join([f"atempo={each:.6f}"] * n)


def pad_silence(audio_path: Path, total_duration: float, position: str = "end", sample_rate: int = 44100) -> None:
    """Pad audio to reach total_duration with silence at the specified position."""
    data, sr = sf.read(audio_path)
    current_duration = len(data) / sr
    if current_duration >= total_duration:
        return
    pad_samples = int((total_duration - current_duration) * sr)
    if data.ndim > 1:
        silence = np.zeros((pad_samples, data.shape[1]), dtype=data.dtype)
    else:
        silence = np.zeros(pad_samples, dtype=data.dtype)
    if position == "end":
        data = np.concatenate([data, silence])
    else:
        data = np.concatenate([silence, data])
    sf.write(audio_path, data, sr)


def fade_out(audio_path: Path, fade_start_sec: float) -> None:
    """Apply a fade-out starting at fade_start_sec."""
    data, sr = sf.read(audio_path)
    # Clamp instead of trusting the caller's value: a negative/zero fade_start_sec
    # (e.g. from a cue with zero-or-negative duration slipping through upstream)
    # would otherwise make fade_len bigger than the data actually sliced below —
    # data[fade_start_sample:] with a negative index is shorter than fade_len,
    # and the multiply crashes with a shape mismatch.
    fade_start_sample = max(0, min(int(fade_start_sec * sr), len(data)))
    if fade_start_sample >= len(data):
        return
    fade_len = len(data) - fade_start_sample
    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    if data.ndim > 1:
        fade = fade[:, np.newaxis]
    data[fade_start_sample:] = data[fade_start_sample:] * fade
    sf.write(audio_path, data, sr)


def mix_audio(
    bg_path: Path, dub_path: Path, output_path: Path,
    bg_volume: float = 0.7, dub_volume: float = 1.0,
) -> None:
    """Mix background audio with dubbed vocals using FFmpeg amix. Output as PCM WAV."""
    filter_complex = (
        f"[0:a]volume={bg_volume}[bg];"
        f"[1:a]volume={dub_volume}[dub];"
        f"[bg][dub]amix=inputs=2:duration=first:normalize=0[mixed]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-i", str(bg_path), "-i", str(dub_path),
        "-filter_complex", filter_complex,
        "-map", "[mixed]", "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
        str(output_path),
    ], capture_output=True, check=True, timeout=300)


def _get_video_codec(video_path: Path) -> str:
    """Detect the video codec of a file using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ], capture_output=True, text=True, errors='replace')
    return result.stdout.strip().lower()


def _nvenc_available() -> bool:
    """Check if h264_nvenc GPU encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, errors='replace', timeout=10,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


_HAS_NVENC = _nvenc_available()


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path, bitrate: str = "192k") -> None:
    """Replace video's audio track with new audio.
    H.264 source → stream copy (instant).
    Non-H.264 → h264_nvenc (GPU) if available, else libx264 (CPU)."""
    codec = _get_video_codec(video_path)
    if codec and codec != "h264":
        if _HAS_NVENC:
            logger.info(f"[MUX] Source codec is {codec}, re-encoding with h264_nvenc (GPU)")
            result = subprocess.run([
                "ffmpeg", "-y", "-hwaccel", "cuda", "-i", str(video_path), "-i", str(audio_path),
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
                "-c:a", "aac", "-b:a", bitrate,
                "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ], capture_output=True, text=True, errors='replace', timeout=600)
        else:
            logger.info(f"[MUX] Source codec is {codec}, re-encoding with libx264 (CPU fallback)")
            result = subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", bitrate,
                "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ], capture_output=True, text=True, errors='replace', timeout=600)
    else:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ], capture_output=True, text=True, errors='replace', timeout=600)
    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        raise subprocess.CalledProcessError(result.returncode, "ffmpeg", result.stdout, result.stderr)


def mux_video_with_mix(
    video_path: Path, bg_path: Path, dub_path: Path, output_path: Path,
    bg_volume: float = 0.7, dub_volume: float = 1.0, bitrate: str = "192k",
) -> None:
    """Mix background + dubbed vocals AND mux with video in a single FFmpeg pass.
    Avoids creating an intermediate WAV file — much faster."""
    filter_complex = (
        f"[1:a]volume={bg_volume}[bg];"
        f"[2:a]volume={dub_volume}[dub];"
        f"[bg][dub]amix=inputs=2:duration=first:normalize=0[mixed]"
    )
    codec = _get_video_codec(video_path)
    if codec and codec != "h264":
        if _HAS_NVENC:
            logger.info(f"[MUX+MIX] Source codec is {codec}, re-encoding with h264_nvenc (GPU)")
            result = subprocess.run([
                "ffmpeg", "-y", "-hwaccel", "cuda",
                "-i", str(video_path), "-i", str(bg_path), "-i", str(dub_path),
                "-filter_complex", filter_complex,
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
                "-c:a", "aac", "-b:a", bitrate,
                "-map", "0:v:0", "-map", "[mixed]",
                "-shortest", "-movflags", "+faststart",
                str(output_path),
            ], capture_output=True, text=True, errors='replace', timeout=600)
        else:
            logger.info(f"[MUX+MIX] Source codec is {codec}, re-encoding with libx264 (CPU)")
            result = subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path), "-i", str(bg_path), "-i", str(dub_path),
                "-filter_complex", filter_complex,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", bitrate,
                "-map", "0:v:0", "-map", "[mixed]",
                "-shortest", "-movflags", "+faststart",
                str(output_path),
            ], capture_output=True, text=True, errors='replace', timeout=600)
    else:
        logger.info(f"[MUX+MIX] Source is {codec or 'h264'}, stream copy")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(bg_path), "-i", str(dub_path),
            "-filter_complex", filter_complex,
            "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
            "-map", "0:v:0", "-map", "[mixed]",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ], capture_output=True, text=True, errors='replace', timeout=600)
    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()
        raise subprocess.CalledProcessError(result.returncode, "ffmpeg", result.stdout, result.stderr)


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """Burn SRT subtitles into video using FFmpeg subtitles filter. Only this step re-encodes video."""
    import shutil
    import tempfile

    # FFmpeg's subtitles filter can't handle colons in paths (e.g. C:\ on Windows).
    # Workaround: copy SRT to a temp dir, use relative path, and set cwd.
    temp_dir = Path(tempfile.mkdtemp())
    temp_srt = temp_dir / "s.srt"
    shutil.copy(str(srt_path), str(temp_srt))

    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "subtitles=s.srt",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ], capture_output=True, text=True, errors='replace', cwd=str(temp_dir))
        if result.returncode != 0:
            if output_path.exists():
                output_path.unlink()
            raise subprocess.CalledProcessError(result.returncode, "ffmpeg", result.stdout, result.stderr)
    finally:
        # Clean up temp dir
        if temp_srt.exists():
            temp_srt.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def srt_to_vtt(srt_path: Path, vtt_path: Path) -> None:
    """Convert SRT subtitle file to WebVTT format."""
    import re
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    vtt_lines = ["WEBVTT\n\n"]
    for line in lines:
        # Only replace commas with periods in timestamp lines (e.g. 00:00:01,000 --> 00:00:04,000)
        if " --> " in line:
            vtt_lines.append(line.replace(",", "."))
        else:
            vtt_lines.append(line)
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.writelines(vtt_lines)


def assemble_dubbed_vocals(
    synced_dir: Path, cues: list[SRTCue], total_duration: float, sample_rate: int = 44100,
    fade_ms: float = 30.0, crossfade_ms: float = 50.0,
    starts: dict[int, float] | None = None,
) -> Path:
    """Assemble all synced cue audio into a single WAV at correct SRT positions.
    Uses crossfade between adjacent cues and smooth fade-in/out per segment.

    With `starts` (cue index -> seconds, from time_sync's elastic timeline),
    each cue is placed there and kept whole: the timeline already guarantees
    cues don't overlap, so nothing is truncated to the SRT slot or crossfaded."""
    total_duration = max(total_duration, 0.1)
    total_samples = int(total_duration * sample_rate)
    mixed = np.zeros(total_samples, dtype=np.float32)
    fade_samples = int(fade_ms / 1000.0 * sample_rate)
    crossfade_samples = int(crossfade_ms / 1000.0 * sample_rate)

    for idx, cue in enumerate(cues):
        synced_path = synced_dir / f"{cue.index:03d}.wav"
        if not synced_path.exists():
            continue
        try:
            data, sr = sf.read(synced_path)
        except Exception:
            continue
        # Convert stereo to mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample if needed
        if sr != sample_rate and sr > 0:
            import torch
            import torchaudio
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            data = resampler(data_tensor).squeeze(0).numpy()
        # Apply fade-in/fade-out for smooth transitions
        if len(data) > 2 * fade_samples:
            data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        # Place audio at the exact cue time slot. Every `mixed` slice below is
        # computed from cue_start + write_len (the *actual* post-truncation
        # length) directly, instead of from separately-derived quantities like
        # end_sample — two numbers meant to agree but computed independently
        # is exactly what caused "operands could not be broadcast together"
        # crashes whenever a cue ran past the end of the track.
        place_at = starts.get(cue.index, cue.start) if starts is not None else cue.start
        cue_start = int(place_at * sample_rate)
        cue_start = max(0, min(cue_start, total_samples))
        if starts is not None:
            write_data = data
        else:
            cue_len = int(cue.end * sample_rate) - cue_start
            write_data = data[:cue_len]
        avail = total_samples - cue_start
        if avail <= 0:
            continue
        write_data = write_data[:avail]
        write_len = len(write_data)
        if write_len == 0:
            continue

        # Check if this cue is close enough to the next for crossfade
        use_crossfade = False
        if starts is None and idx < len(cues) - 1 and crossfade_samples > 0:
            gap_to_next = cues[idx + 1].start - cue.end
            if gap_to_next < 0.3:  # Only crossfade if gap is small (<300ms)
                use_crossfade = True

        overlap = min(crossfade_samples, write_len // 2) if use_crossfade else 0
        if overlap > 0:
            fade_out_curve = np.linspace(1, 0, overlap)
            write_data[-overlap:] *= fade_out_curve
        head_len = write_len - overlap
        mixed[cue_start:cue_start + head_len] += write_data[:head_len]
        if overlap > 0:
            mixed[cue_start + head_len:cue_start + write_len] += write_data[head_len:]

    # Normalize to prevent clipping from additive overlaps
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    output_path = synced_dir.parent / "dubbed_vocals.wav"
    sf.write(output_path, mixed, sample_rate)
    return output_path


# --- Project helpers ---

def generate_project_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_project_dirs(project_dir: Path) -> None:
    (project_dir / "input").mkdir(parents=True, exist_ok=True)
    (project_dir / "work").mkdir(parents=True, exist_ok=True)
    (project_dir / "work" / "pt_segments").mkdir(parents=True, exist_ok=True)
    (project_dir / "work" / "pt_synced").mkdir(parents=True, exist_ok=True)
    (project_dir / "output").mkdir(parents=True, exist_ok=True)