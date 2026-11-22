FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential python3-pip libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install --pre torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall

WORKDIR /app

RUN pip install --no-cache-dir \
    "faster-whisper>=1.0.0" \
    "transformers>=4.50.0" \
    "sentencepiece>=0.2.0" \
    "sacremoses>=0.1.0"

RUN pip install --no-cache-dir "demucs>=4.0.0"

RUN pip install --no-cache-dir \
    "qwen-tts==0.1.1" "transformers==4.57.3" \
    "edge-tts>=6.0.0"

RUN pip uninstall torchvision -y 2>/dev/null || true

RUN pip install --no-cache-dir torchcodec \
    --index-url https://download.pytorch.org/whl/nightly/cu128


RUN pip install --no-cache-dir --ignore-installed \
    "blinker>=1.9.0" "flask>=2.0.0" "flask-cors>=3.0.0" \
    "waitress>=2.1.0" "gunicorn>=21.0.0" "gevent>=23.0.0"

RUN pip install --no-cache-dir \
    "soundfile>=0.12.0" "numpy>=1.24.0,<2.0.0" \
    "audiostretchy>=1.2.0" "psutil>=5.9.0" "pysubs2>=1.7.0"

RUN pip install --no-cache-dir "gradio==4.44.1" || echo "Gradio nao instalado"

COPY requirements.txt .
COPY . .

ENV HF_HUB_DISABLE_SYMLINKS=1
ENV PYTHONUNBUFFERED=1
# Detecta a GPU do pod e escolhe o perfil sozinho (config.py)
ENV DUB_PROFILE=auto

VOLUME ["/app/data"]
EXPOSE 5000
CMD ["python3", "server.py"]
