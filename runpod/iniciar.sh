#!/usr/bin/env bash
# =====================================================================
#  LuminaDub AI - RunPod
#
#  Um comando so: instala (so na primeira vez), baixa os modelos e liga
#  o servidor em segundo plano. Funciona em qualquer GPU NVIDIA.
#  Pod com varias placas: liga um servidor por placa (portas 5000, 5001...),
#  cada um com a propria fila. DUB_GPUS=1 forca usar so a primeira.
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
BASE_PORT=5000

# Quantas placas usar: todas do pod (ou DUB_GPUS).
gpu_count() {
  local n=0
  command -v nvidia-smi >/dev/null 2>&1 && n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
  [ "${n:-0}" -ge 1 ] || n=1
  if [ -n "${DUB_GPUS:-}" ] && [ "$DUB_GPUS" -ge 1 ] 2>/dev/null && [ "$DUB_GPUS" -lt "$n" ]; then n="$DUB_GPUS"; fi
  echo "$n"
}
NGPU="$(gpu_count)"
log_of()  { echo "$WS/luminadub-gpu$1.log"; }
pid_of()  { echo "$WS/luminadub-gpu$1.pid"; }
port_of() { echo $((BASE_PORT + $1)); }

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

url() {  # url <porta>
  if [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "https://${RUNPOD_POD_ID}-$1.proxy.runpod.net"
  else
    echo "http://localhost:$1"
  fi
}

running() { local f; f="$(pid_of "$1")"; [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; }
any_running() { local i; for i in $(seq 0 9); do running "$i" && return 0; done; return 1; }
profile_of() { grep -o "perfil automatico '[a-z0-9]*'" "$(log_of "$1")" 2>/dev/null | tail -1 || true; }

gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
  fi
}

cmd_status() {
  banner
  local g; g="$(gpu_info || true)"
  [ -n "$g" ] && ok "GPU: $g$( [ "$NGPU" -gt 1 ] && echo " (x$NGPU)")" || warn "Nenhuma GPU NVIDIA encontrada"
  local i found=0
  for i in $(seq 0 9); do
    running "$i" || continue
    found=1
    ok "Placa $((i+1)): ligada  ->  ${B}$(url "$(port_of "$i")")${N}  $(profile_of "$i")"
  done
  [ "$found" = 1 ] || warn "Servidor desligado. Ligue com: bash runpod/iniciar.sh"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
      | while IFS=, read -r idx used total util; do info "Placa $((idx+1)): VRAM$used de$total, uso$util"; done
  fi
}

cmd_parar() {
  local i stopped=0 f
  for i in $(seq 0 9); do
    f="$(pid_of "$i")"
    if [ -f "$f" ]; then
      # Primeiro o laço de reinício (senão ele religa o servidor), depois o Python.
      local pid; pid="$(cat "$f")"
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      rm -f "$f"; stopped=1
    fi
  done
  pkill -f "venv-luminadub/bin/python server.py" 2>/dev/null || true
  [ "$stopped" = 1 ] && ok "Servidor desligado." || warn "O servidor já estava desligado."
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
  step 4/4 "Ligando $( [ "$NGPU" -gt 1 ] && echo "$NGPU servidores (um por placa)" || echo "o servidor")"
  if any_running; then
    warn "Já estava ligado — reiniciando para pegar o código novo"
    cmd_parar >/dev/null
    sleep 2
  fi
  local i
  for i in $(seq 0 $((NGPU - 1))); do
    local port; port="$(port_of "$i")"
    local env="DUB_PORT=$port"
    if [ "$NGPU" -gt 1 ]; then
      # Cada servidor enxerga só a sua placa e tem a sua própria fila.
      env="$env CUDA_VISIBLE_DEVICES=$i"
      [ "$i" -gt 0 ] && env="$env DUB_PROJECTS_DIR='$APP_DIR/data/projects-gpu$i'"
    fi
    # Laço de reinício: se o servidor cair, volta sozinho em 5 s (igual ao start.bat).
    # setsid: laço + servidor num grupo próprio, assim "parar" derruba os dois juntos.
    nohup setsid bash -c "
      cd '$APP_DIR'
      export $env
      while true; do
        '$VENV/bin/python' server.py
        code=\$?
        [ \$code -eq 0 ] && break
        echo \"[RUNPOD] Servidor caiu (codigo \$code). Reiniciando em 5 s...\"
        sleep 5
      done
    " >> "$(log_of "$i")" 2>&1 &
    echo $! > "$(pid_of "$i")"
  done

  info "Esperando $( [ "$NGPU" -gt 1 ] && echo "os servidores responderem" || echo "o servidor responder")..."
  local pending
  for _ in $(seq 1 90); do
    pending=0
    for i in $(seq 0 $((NGPU - 1))); do
      curl -s -o /dev/null "http://127.0.0.1:$(port_of "$i")/" || pending=1
    done
    if [ "$pending" = 0 ]; then
      echo
      ok "${B}Pronto!${N}"
      echo
      for i in $(seq 0 $((NGPU - 1))); do
        local label="Abra no navegador:"
        [ "$NGPU" -gt 1 ] && label="Placa $((i+1)):           "
        echo -e "  ${B}${label}${N}  ${G}$(url "$(port_of "$i")")${N}  $(profile_of "$i")"
      done
      if [ "$NGPU" -gt 1 ]; then
        echo
        echo "  Cada link tem a sua fila: mande uma parte dos cursos em cada um."
        echo "  Os dublados de todos vao pra mesma pasta: data/dublados/<Curso>/..."
        echo "  (No RunPod, exponha as portas $BASE_PORT a $(port_of $((NGPU - 1))) em Expose HTTP Ports.)"
      fi
      echo
      echo "  Comandos úteis:"
      echo "    bash runpod/iniciar.sh status   → ver placas, links e VRAM"
      echo "    bash runpod/iniciar.sh logs     → acompanhar a dublagem"
      echo "    bash runpod/iniciar.sh parar    → desligar"
      echo
      echo -e "  ${Y}Lembre de dar Stop no pod quando terminar — ele cobra por hora ligado.${N}"
      return 0
    fi
    sleep 2
  done
  fail "Não respondeu em 3 minutos. Veja o erro com: tail -50 $WS/luminadub-gpu0.log"
}

case "${1:-}" in
  status)     cmd_status ;;
  logs)       tail -n 30 -f "$WS"/luminadub-gpu*.log ;;
  parar)      cmd_parar ;;
  reinstalar) cmd_parar || true; rm -rf "$VENV"; banner; instalar; ligar ;;
  ""|iniciar) banner; instalar; ligar ;;
  *) echo "Uso: bash runpod/iniciar.sh [status|logs|parar|reinstalar]"; exit 1 ;;
esac
