"""Baixa todos os modelos para o volume, usando os MESMOS nomes do config.py
(assim nunca fica desatualizado se você trocar de modelo).

Uso: python runpod/baixar_modelos.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402

cfg = Config()
cfg.ensure_dirs()

print(f"[1/4] Qwen3-TTS ({cfg.QWEN_TTS_MODEL})...")
qwen_dir = cfg.MODEL_CACHE_DIR / "qwen3-tts" / "Qwen3-TTS-12Hz-1.7B-Base"
if not (qwen_dir / "config.json").exists():
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir=str(qwen_dir))
print("      OK")

print(f"[2/4] Whisper {cfg.WHISPER_MODEL}...")
from faster_whisper.utils import download_model  # noqa: E402
download_model(cfg.WHISPER_MODEL, cache_dir=str(cfg.MODEL_CACHE_DIR / "whisper"))
print("      OK")

print(f"[3/4] Demucs {cfg.DEMUCS_MODEL}...")
from demucs.pretrained import get_model  # noqa: E402
get_model(cfg.DEMUCS_MODEL)
print("      OK")

print(f"[4/4] Tradutor {cfg.TRANSLATION_MODEL}...")
from transformers import MarianMTModel, MarianTokenizer  # noqa: E402
MarianTokenizer.from_pretrained(cfg.TRANSLATION_MODEL)
MarianMTModel.from_pretrained(cfg.TRANSLATION_MODEL)
print("      OK")

print("Todos os modelos prontos.")
