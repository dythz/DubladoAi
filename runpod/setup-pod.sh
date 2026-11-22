#!/bin/bash
# Run this INSIDE the RunPod container after it starts
# Pre-downloads all ML models so first dubbing is fast

echo "========================================="
echo "  LuminaDub AI - Download de Modelos"
echo "========================================="
echo ""

cd /app

# Set environment variables before loading ML libs
export HF_HUB_DISABLE_SYMLINKS=1

python3 -c "
import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

print('[1/4] Baixando Whisper large-v3-turbo...')
from faster_whisper import WhisperModel
WhisperModel('large-v3-turbo', device='cuda', compute_type='float16', download_root='/app/data/models/whisper')
print('  OK')

print('[2/4] Baixando Demucs htdemucs...')
from demucs.pretrained import get_model
get_model('htdemucs')
print('  OK')

print('[3/4] Baixando Opus-MT EN->PT...')
from transformers import MarianMTModel, MarianTokenizer
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-pt')
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-big-en-pt')
print('  OK')

print('[4/4] Baixando Qwen3-TTS 1.7B Base...')
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='/app/data/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base')
print('  OK')
"

echo ""
echo "========================================="
echo "  Todos os modelos baixados com sucesso"
echo "========================================="

# Perfil da placa detectado sozinho (config.py -> detect_profile)
export DUB_PROFILE=auto
echo "Iniciando o servidor na porta 5000..."
exec python3 /app/server.py
