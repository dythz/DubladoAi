#!/usr/bin/env bash
# =====================================================================
#  LuminaDub AI - RunPod
#
#  Um comando so: instala (so na primeira vez), baixa os modelos e liga
#  o servidor em segundo plano. Funciona em qualquer GPU NVIDIA.
#
#    bash runpod/iniciar.sh            instala (se precisar) e liga
#    bash runpod/iniciar.sh status     mostra GPU, servidor e link
#    bash runpod/iniciar.sh logs       acompanha o log ao vivo
#    bash runpod/iniciar.sh parar      desliga o servidor
#    bash runpod/iniciar.sh reinstalar apaga o ambiente e instala de novo
# =====================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${LUMINADUB_WORKSPACE:-/workspace}"
[ -d "$WS" ] || WS="$APP_DIR"
VENV="$WS/venv-luminadub"
LOG="$WS/luminadub.log"
PIDFILE="$WS/luminadub.pid"
PORT=5000

# Tudo que é baixado fica no volume: religar o pod não baixa nada de novo.
export HF_HOME="$WS/cache/huggingface"
export TORCH_HOME="$WS/cache/torch"
export PIP_CACHE_DIR="$WS/cache/pip"
export HF_HUB_DISABLE_SYMLINKS=1
export PYTHONUNBUFFERED=1
export DUB_PROFILE="${DUB_PROFILE:-auto}"

G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; C='\033[1;36m'; B='\033[1m'; N='\033[0m'
ok()   { echo -e "  ${G}✔${N} $*"; }
info() { echo -e "  ${C}➜${N} $*"; }
warn() { echo -e "  ${Y}!${N} $*"; }
fail() { echo -e "  ${R}✖ $*${N}"; exit 1; }
step() { echo -e "\n${B}[$1]${N} $2"; }

banner() {
  echo -e "${C}"
  echo "  ╔══════════════════════════════════════╗"
  echo "  ║          LuminaDub AI · RunPod       ║"
  echo "  ╚══════════════════════════════════════╝"
  echo -e "${N}"
}

url() {
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
  else
    echo "http://localhost:${PORT}"
  fi
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
  fi
}

cmd_status() {
  banner
  local g; g="$(gpu_info || true)"
  [ -n "$g" ] && ok "GPU: $g" || warn "Nenhuma GPU NVIDIA encontrada"
  if running; then
    ok "Servidor ligado (PID $(cat "$PIDFILE"))"
    info "Abra: ${B}$(url)${N}"
    local prof; prof="$(grep -o "perfil automatico '[a-z0-9]*'" "$LOG" 2>/dev/null | tail -1 || true)"
    [ -n "$prof" ] && info "Configuração: $prof"
  else
    warn "Servidor desligado. Ligue com: bash runpod/iniciar.sh"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    info "VRAM em uso: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
  fi
}

cmd_parar() {
  if running; then
    # Primeiro o laço de reinício (senão ele religa o servidor), depois o Python.
    local pid; pid="$(cat "$PIDFILE")"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    ok "Servidor desligado."
  else
    warn "O servidor já estava desligado."
  fi
  pkill -f "venv-luminadub/bin/python server.py" 2>/dev/null || true
}

# ---------------------------------------------------------------------
instalar() {
  step 1/4 "Pacotes do sistema"
  local need=()
  command -v ffmpeg >/dev/null 2>&1 || need+=(ffmpeg)
  ldconfig -p 2>/dev/null | grep -q libsndfile || need+=(libsndfile1)
  command -v curl >/dev/null 2>&1 || need+=(curl)
  python3 -c "import ensurepip" 2>/dev/null || need+=(python3-venv)
  if [ ${#need[@]} -gt 0 ]; then
    info "Instalando: ${need[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq "${need[@]}" >/dev/null
  fi
  ok "ffmpeg $(ffmpeg -version | head -1 | awk '{print $3}')"

  step 2/4 "Ambiente Python (fica salvo no volume)"
  local req_hash; req_hash="$(sha1sum "$APP_DIR/requirements.txt" | cut -c1-12)"
  if [ -f "$VENV/.instalado" ] && [ "$(cat "$VENV/.instalado")" = "$req_hash" ]; then
    ok "Já instalado — pulando (requirements.txt não mudou)"
  else
    [ -d "$VENV" ] || python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install -q --upgrade pip wheel

    # Placas Blackwell (RTX 50xx, B200, B300, PRO 6000) precisam do PyTorch
    # com CUDA 12.8. As outras usam o 12.6, igual ao seu PC.
    local cc; cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
    local cu="cu126"; [ "${cc:-0}" -ge 10 ] && cu="cu128"
    info "PyTorch 2.7.1 ($cu) — download grande, ~2.5 GB..."
    "$VENV/bin/pip" install -q torch==2.7.1 torchaudio==2.7.1 --index-url "https://download.pytorch.org/whl/$cu"

    info "Bibliotecas do requirements.txt..."
    # deepfilternet é opcional (limpa o áudio de referência); se não tiver
    # versão pra este Python, o resto instala igual e o programa segue sem ele.
    grep -v -i "^deepfilternet" "$APP_DIR/requirements.txt" > /tmp/req-luminadub.txt
    "$VENV/bin/pip" install -q -r /tmp/req-luminadub.txt
    if "$VENV/bin/pip" install -q "deepfilternet>=0.5.6" 2>/dev/null; then
      ok "deepfilternet instalado"
    else
      warn "deepfilternet não instalou — a voz de referência não será limpa (resto funciona)"
    fi
    echo "$req_hash" > "$VENV/.instalado"
    ok "Ambiente pronto"
  fi

  "$VENV/bin/python" - <<'PY' || fail "PyTorch não enxerga a GPU. Confira se o pod tem GPU NVIDIA."
import torch, sys
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
print(f"  \033[1;32m✔\033[0m PyTorch {torch.__version__} vê a {p.name} ({p.total_memory/1024**3:.0f} GB)")
try:
    import faster_qwen3_tts  # noqa
    print("  \033[1;32m✔\033[0m faster-qwen3-tts (voz ~2x mais rápida) ativo")
except Exception as e:
    print(f"  \033[1;33m!\033[0m faster-qwen3-tts indisponível ({e}) — voz vai no modo normal")
PY

  step 3/4 "Modelos (só baixa na primeira vez)"
  (cd "$APP_DIR" && "$VENV/bin/python" runpod/baixar_modelos.py 2>&1 | sed 's/^/  /')
}

ligar() {
  step 4/4 "Ligando o servidor"
  if running; then
    warn "Já estava ligado — reiniciando para pegar o código novo"
    cmd_parar >/dev/null
    sleep 2
  fi
  # Laço de reinício: se o servidor cair, volta sozinho em 5 s (igual ao start.bat).
  # setsid: laço + servidor num grupo próprio, assim "parar" derruba os dois juntos.
  nohup setsid bash -c "
    cd '$APP_DIR'
    while true; do
      '$VENV/bin/python' server.py
      code=\$?
      [ \$code -eq 0 ] && break
      echo \"[RUNPOD] Servidor caiu (codigo \$code). Reiniciando em 5 s...\"
      sleep 5
    done
  " >> "$LOG" 2>&1 &
  echo $! > "$PIDFILE"

  info "Esperando o servidor responder..."
  for _ in $(seq 1 90); do
    if curl -s -o /dev/null "http://127.0.0.1:${PORT}/"; then
      echo
      ok "${B}Pronto!${N}"
      echo
      echo -e "  ${B}Abra no navegador:${N}  ${G}$(url)${N}"
      local prof; prof="$(grep -o "perfil automatico '[a-z0-9]*'" "$LOG" | tail -1 || true)"
      [ -n "$prof" ] && echo -e "  ${B}Configuração:${N}       $prof"
      echo
      echo "  Comandos úteis:"
      echo "    bash runpod/iniciar.sh status   → ver GPU, link e VRAM"
      echo "    bash runpod/iniciar.sh logs     → acompanhar a dublagem"
      echo "    bash runpod/iniciar.sh parar    → desligar o servidor"
      echo
      echo -e "  ${Y}Lembre de dar Stop no pod quando terminar — ele cobra por hora ligado.${N}"
      return 0
    fi
    sleep 2
  done
  fail "O servidor não respondeu em 3 minutos. Veja o erro com: tail -50 $LOG"
}

case "${1:-}" in
  status)     cmd_status ;;
  logs)       tail -n 50 -f "$LOG" ;;
  parar)      cmd_parar ;;
  reinstalar) cmd_parar || true; rm -rf "$VENV"; banner; instalar; ligar ;;
  ""|iniciar) banner; instalar; ligar ;;
  *) echo "Uso: bash runpod/iniciar.sh [status|logs|parar|reinstalar]"; exit 1 ;;
esac
