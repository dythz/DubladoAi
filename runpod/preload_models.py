import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

print("[PRELOAD] Baixando modelos...")

print("[PRELOAD] 1/3 Whisper large-v3-turbo...")
from faster_whisper import WhisperModel
WhisperModel("large-v3-turbo", device="cuda", compute_type="float16", download_root="/app/data/models/whisper")
print("[PRELOAD] Whisper ok")

print("[PRELOAD] 2/3 opus-mt...")
from transformers import MarianMTModel, MarianTokenizer
MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt", cache_dir="/app/data/models/opus")
MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-pt", cache_dir="/app/data/models/opus")
print("[PRELOAD] opus-mt ok")

print("[PRELOAD] 3/3 Qwen3-TTS 1.7B Base...")
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="/app/data/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base")
print("[PRELOAD] Qwen3-TTS ok")

print("[PRELOAD] Todos os modelos prontos!")
