import os
from pathlib import Path
from dataclasses import dataclass, field

_BASE = Path(os.environ.get("APP_BASE_DIR", str(Path(__file__).parent)))
_QWEN_LOCAL = str(_BASE / "data" / "models" / "qwen3-tts" / "Qwen3-TTS-12Hz-1.7B-Base")

@dataclass
class Config:
    BASE_DIR: Path = _BASE
    PROJECTS_DIR: Path = _BASE / "data" / "projects"
    MODEL_CACHE_DIR: Path = _BASE / "data" / "models"

    # GPU
    DEVICE: str = "cuda"
    MAX_VRAM_MB: int = 90000
    # Drop models the current stage doesn't use from VRAM (Qwen stays warm).
    EVICT_IDLE_MODELS: bool = False
    # Kinds (model_cache key[0]) never evicted, even with EVICT_IDLE_MODELS.
    ALWAYS_WARM_MODELS: list = field(default_factory=lambda: ["qwen3-tts", "qwen3-accent"])

    # --- Velocidade ---
    # Carrega o Qwen em segundo plano enquanto Demucs/Whisper/tradução rodam,
    # em vez de esperar a etapa de voz começar (~60 s a menos no 1º vídeo).
    QWEN_PRELOAD: bool = True
    # Quantas falas o Qwen gera ao mesmo tempo (cada uma num modelo próprio na
    # VRAM). 1 = uma por vez. Só vale com o backend "faster".
    QWEN_PARALLEL: int = 1
    # Pula o Demucs quando o fundo é silêncio (aula só com narração): mede o
    # volume nas pausas da fala; abaixo deste nível (dBFS) não há música.
    AUTO_SKIP_SEPARATION: bool = True
    SEPARATION_SILENCE_DB: float = -45.0
    # Enquanto um vídeo gera a voz, o próximo da fila já faz extrair, separar,
    # transcrever e traduzir. Só em placas com VRAM sobrando.
    PIPELINE_OVERLAP: bool = False
    # Depois que o vídeo fica pronto, apaga os áudios intermediários (~200 MB
    # por vídeo; num curso de 30 h passaria de 90 GB).
    CLEAN_WORK_AFTER_DONE: bool = True

    # Demucs
    # htdemucs_ft (4-model fine-tuned ensemble) separates cleaner but is ~4x slower.
    # Defaulting back to the fast single-model htdemucs; switch to htdemucs_ft if
    # separation quality becomes the bottleneck instead of speed.
    DEMUCS_MODEL: str = "htdemucs"
    DEMUCS_TWO_STEM: str = "vocals"
    DEMUCS_SEGMENT: float = 40.0
    # shifts=N re-runs inference N times on shifted copies of the audio and averages
    # the result — a small quality bump for a roughly Nx runtime cost. 1 = no shifting.
    DEMUCS_SHIFTS: int = 1
    DEMUCS_OVERLAP: float = 0.25

    # Whisper
    # large-v3-turbo: ~4x faster than large-v3 with only a small accuracy tradeoff
    # (fewer decoder layers) — worth it since transcription doesn't affect voice quality.
    WHISPER_MODEL: str = "large-v3-turbo"
    WHISPER_COMPUTE_TYPE: str = "float16"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_LANGUAGE: str = "en"
    WHISPER_VAD_FILTER: bool = True
    WHISPER_BEAM_SIZE: int = 5
    # >1 = modo em lote do faster-whisper (vários trechos ao mesmo tempo).
    # 1 = sequencial, igual sempre foi.
    WHISPER_BATCH_SIZE: int = 1
    WORDS_PER_CUE: int = 7
    MAX_WORDS_PER_CUE: int = 10
    MAX_CHARS_PER_CUE: int = 45
    MIN_CUE_DURATION: float = 1.0
    MIN_CUE_GAP: float = 0.08
    MAX_CUE_DURATION: float = 10.0

    # Translation
    TRANSLATION_ENGINE: str = "opus-mt"
    TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    OPUS_TARGET_TOKEN: str = ">>pob<<"  # Brazilian Portuguese (">>por<<" = European)
    TRANSLATION_DEVICE: str = "cuda"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_URL: str = "http://localhost:11434"
    # Ask Ollama to drop its model from VRAM once translation finishes, so the
    # LLM isn't still sitting on the GPU while TTS needs it.
    OLLAMA_UNLOAD_AFTER: bool = False

    # TTS
    TTS_DEFAULT: str = "qwen"
    # Qwen3-TTS: natural voice cloning (Apache 2.0). Downloaded into data/models
    # by the setup step; falls back to the Hugging Face repo id if it isn't there.
    QWEN_TTS_MODEL: str = _QWEN_LOCAL if Path(_QWEN_LOCAL).exists() else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    QWEN_LANGUAGE: str = "portuguese"
    # "faster" = faster-qwen3-tts (CUDA graphs, ~2x faster); "transformers" = plain qwen-tts.
    QWEN_BACKEND: str = "faster"
    QWEN_BATCH_SIZE: int = 8
    # Also cap total characters per batch (VRAM scales with output length).
    QWEN_BATCH_CHARS: int = 10000
    QWEN_MAX_RETRIES: int = 2
    # Sampling temperature. The checkpoint default (0.9) lets each cue wander in
    # timbre/delivery; lower keeps one consistent voice across the whole dub.
    QWEN_TEMPERATURE: float = 0.7
    # Takes of the pt-BR example said in the target voice; the closest one
    # becomes the ICL context for every cue (0 = use the stock example as-is).
    QWEN_ANCHOR_TRIES: int = 4
    # Short pt-BR clips used as the speaking-style example when cloning a voice
    # that was recorded in another language (e.g. the course's English
    # narrator): the timbre still comes from that voice, the accent from these.
    # Off: the dub uses only the video's own voice (speaker embedding from the
    # cleanest narrator clip). Measured on a real lesson this is the most
    # consistent and closest to the narrator. On: borrow pt-BR delivery from the
    # example clips below — better accent, but lines drift towards the
    # example speaker's timbre and the voice audibly changes between lines.
    QWEN_BR_ACCENT_EXAMPLE: bool = False
    QWEN_ACCENT_DIR: str = str(_BASE / "data" / "models" / "qwen3-tts" / "sotaque_br")
    EDGE_TTS_VOICE: str = "pt-BR-ThalitaMultilingualNeural"
    EDGE_TTS_RATE: str = "+0%"

    # Reference audio enhancement (denoise the cloning clip if it's noisy)
    ENHANCE_REFERENCE_AUDIO: bool = True

    # Time sync
    MAX_SPEED_RATIO: float = 1.5
    # Slowing speech much below this sounds drawn-out/"moaning"; short cues are
    # padded with silence instead.
    MIN_SPEED_RATIO: float = 0.90
    STRETCH_METHOD: str = "atempo"
    TOLERANCE_RATIO: float = 0.08
    # Cues are never cut. A cue that still doesn't fit at MAX_SPEED_RATIO runs
    # into the next one's slot and pushes it later; once the dub is more than
    # SYNC_CATCHUP_AFTER_SEC behind the original, cues may speed up to
    # SYNC_CATCHUP_SPEED to win that time back.
    SYNC_CATCHUP_SPEED: float = 1.75
    SYNC_CATCHUP_AFTER_SEC: float = 0.5
    SYNC_MIN_GAP_SEC: float = 0.05
    # Safety valve: a cue that would leave the dub more than this far behind
    # (only happens with broken/babbling TTS output) is trimmed so one bad cue
    # can't push the rest of the video out of sync.
    SYNC_MAX_DRIFT_SEC: float = 3.0

    # Audio assembly
    FADE_MS: float = 30.0
    CROSSFADE_MS: float = 50.0

    # Merge
    BACKGROUND_VOLUME: float = 0.7
    DUB_VOLUME: float = 1.0
    AUDIO_BITRATE: str = "192k"

    # Pipeline
    STAGE_ORDER: list = field(default_factory=lambda: [
        "extract", "separate", "transcribe", "translate",
        "synthesize", "sync", "merge"
    ])
    STAGE_WEIGHTS: dict = field(default_factory=lambda: {
        "extract": 0.02, "separate": 0.25, "transcribe": 0.15,
        "translate": 0.08, "synthesize": 0.30, "sync": 0.10, "merge": 0.10
    })
    STAGE_NAMES: dict = field(default_factory=lambda: {
        "extract": "Extract Audio",
        "separate": "Vocal Separation",
        "transcribe": "Transcribe (Whisper)",
        "translate": "Translate EN→PT",
        "synthesize": "Synthesize (TTS)",
        "sync": "Time Sync",
        "merge": "Merge Video"
    })

    enable_vocal_separation: bool = True
    tts_engine: str = "qwen"
    reference_audio_path: str = ""
    reference_text: str = ""
    reference_language: str = ""

    def __post_init__(self):
        # DUB_PROFILE vazio ou "auto" = detecta a placa sozinho. Um nome de
        # PROFILES (ex.: "rtx3060") força aquele perfil.
        profile = os.environ.get("DUB_PROFILE", "").strip().lower() or "auto"
        if profile == "auto":
            profile, vram_mb, gpu_name = detect_profile()
            if vram_mb:
                self.MAX_VRAM_MB = max(vram_mb - 512, 1024)
            _log_profile_once(profile, gpu_name, vram_mb, auto=True)
        elif profile not in PROFILES:
            raise ValueError(f"Unknown DUB_PROFILE '{profile}'. Options: auto, {', '.join(PROFILES)}")
        else:
            _log_profile_once(profile, None, None, auto=False)
        for key, val in PROFILES[profile].items():
            setattr(self, key, val)
        # Ajuste fino sem mexer no código (ex.: no start.bat):
        #   set "DUB_QWEN_PARALLEL=2"  -> 2 falas ao mesmo tempo
        par = os.environ.get("DUB_QWEN_PARALLEL", "").strip()
        if par.isdigit() and int(par) >= 1:
            self.QWEN_PARALLEL = int(par)

    def ensure_dirs(self):
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Hardware profiles, picked with the DUB_PROFILE env var (start.bat sets it).
# Only keys the web UI doesn't send per job belong here — anything the UI sends
# (tts_engine, translation engine, cue sizes...) overrides the profile anyway.
PROFILES: dict[str, dict] = {
    # RTX 3060 12GB, local. All models cached together plus Qwen generating 8
    # long cues at once went past 12GB and spilled into (slow) system RAM, so
    # idle models are freed per stage and Qwen batches are capped by text size.
    "rtx3060": {
        "MAX_VRAM_MB": 11500,
        "EVICT_IDLE_MODELS": True,
        "QWEN_BATCH_CHARS": 360,
        # int8 weights with fp16 compute: ~half the VRAM of float16 and faster
        # on Ampere, with no audible accuracy loss on clean course audio.
        "WHISPER_COMPUTE_TYPE": "int8_float16",
        # Greedy decoding: ~2x faster than beam 5; course narration is clean
        # studio speech, where beam search buys almost nothing.
        "WHISPER_BEAM_SIZE": 1,
        # A 7-8B Ollama model takes ~5GB — free it before TTS loads.
        "OLLAMA_UNLOAD_AFTER": True,
        # Demucs (~0.3GB) e Opus-MT (~1GB) são pequenos: ficam carregados entre
        # vídeos em vez de recarregar (~30 s) a cada um. Whisper ainda sai.
        "ALWAYS_WARM_MODELS": ["qwen3-tts", "qwen3-accent", "demucs", "translation"],
        "QWEN_PARALLEL": 1,
        "PIPELINE_OVERLAP": False,
    },
    # 16-24GB (RTX 4080/4090/3090, A5000, L4). Whisper + Demucs + Opus-MT +
    # Qwen cabem juntos, então nada é descarregado entre etapas (sem recarregar
    # modelo = mais rápido). Ollama ainda sai antes do TTS por segurança.
    "gpu24": {
        "EVICT_IDLE_MODELS": False,
        "QWEN_BATCH_CHARS": 2000,
        "WHISPER_COMPUTE_TYPE": "float16",
        "WHISPER_BEAM_SIZE": 1,
        "OLLAMA_UNLOAD_AFTER": True,
        # 2 falas ao mesmo tempo (~4-5GB de VRAM por cópia do Qwen).
        "QWEN_PARALLEL": 2,
        "PIPELINE_OVERLAP": True,
        "WHISPER_BATCH_SIZE": 8,
    },
    # 32GB ou mais (RTX 5090, A6000, L40S, RTX 6000 Ada, A100, H100...).
    # Tudo fica carregado o tempo todo, inclusive o Ollama.
    "gpu48": {
        "EVICT_IDLE_MODELS": False,
        "QWEN_BATCH_CHARS": 10000,
        "WHISPER_COMPUTE_TYPE": "float16",
        "WHISPER_BEAM_SIZE": 1,
        "OLLAMA_UNLOAD_AFTER": False,
        "QWEN_PARALLEL": 3,
        "PIPELINE_OVERLAP": True,
        "WHISPER_BATCH_SIZE": 16,
    },
    # Sem GPU NVIDIA: roda na CPU (muito lento, só pra não quebrar).
    "cpu": {
        "DEVICE": "cpu",
        "WHISPER_DEVICE": "cpu",
        "WHISPER_COMPUTE_TYPE": "int8",
        "WHISPER_BEAM_SIZE": 1,
        "TRANSLATION_DEVICE": "cpu",
        "QWEN_BACKEND": "transformers",
        "QWEN_BATCH_CHARS": 360,
        "EVICT_IDLE_MODELS": True,
        "QWEN_PRELOAD": False,
        "QWEN_PARALLEL": 1,
    },
}


_detected: tuple | None = None
_logged = False


def detect_profile() -> tuple[str, int, str]:
    """Olha a GPU 0 e escolhe o perfil pela VRAM. Retorna (perfil, vram_mb, nome).
    Resultado guardado: Config() é criado várias vezes por processo."""
    global _detected
    if _detected is not None:
        return _detected
    try:
        import torch
        if not torch.cuda.is_available():
            _detected = ("cpu", 0, "CPU")
            return _detected
        props = torch.cuda.get_device_properties(0)
        vram_mb = props.total_memory // (1024 * 1024)
        name = props.name
    except Exception:
        _detected = ("cpu", 0, "CPU")
        return _detected
    # Margens: uma "12GB" reporta ~12288MB, uma "24GB" ~24564MB, etc.
    if vram_mb < 14000:
        profile = "rtx3060"
    elif vram_mb < 30000:
        profile = "gpu24"
    else:
        profile = "gpu48"
    _detected = (profile, vram_mb, name)
    return _detected


def _log_profile_once(profile: str, gpu_name, vram_mb, auto: bool) -> None:
    global _logged
    if _logged:
        return
    _logged = True
    import logging
    log = logging.getLogger("dubbing")
    if auto:
        log.info(f"[GPU] {gpu_name} ({(vram_mb or 0) / 1024:.0f} GB) -> perfil automatico '{profile}'")
        if profile == "cpu":
            log.warning("[GPU] Nenhuma GPU NVIDIA encontrada: rodando na CPU (vai ser bem lento)")
    else:
        log.info(f"[GPU] Perfil forcado pelo DUB_PROFILE: '{profile}'")
