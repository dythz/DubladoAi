# Imagem pronta (opcional). O jeito mais fácil é o template PyTorch do RunPod
# + runpod/iniciar.sh (veja GUIA.md). Use esta imagem só se quiser um pod que
# já liga com tudo instalado.
#
#   docker build -f runpod/Dockerfile -t seunome/luminadub-ai .
#
# PyTorch 2.7.1 com CUDA 12.8: o mesmo PyTorch do PC e roda de Ampere (RTX 30xx,
# A6000) até Blackwell (RTX 50xx, B200).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS=1 \
    DUB_PROFILE=auto \
    HF_HOME=/app/data/cache/huggingface \
    TORCH_HOME=/app/data/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependências antes do código: mudar só o código não reinstala tudo.
COPY requirements.txt .
RUN grep -v -i "^deepfilternet" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && (pip install --no-cache-dir "deepfilternet>=0.5.6" || echo "deepfilternet indisponivel - seguindo sem ele")

COPY . .

# Modelos, vozes e projetos: monte um volume em /app/data.
VOLUME ["/app/data"]
EXPOSE 5000
CMD ["bash", "-c", "python runpod/baixar_modelos.py && exec python server.py"]
