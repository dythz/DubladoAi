@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title LuminaDub AI - Instalacao
echo.
echo  +======================================+
echo  ^|      LuminaDub AI - Instalacao       ^|
echo  +======================================+
echo.
echo  Isso prepara tudo para rodar: ambiente Python, bibliotecas
echo  e o modelo de voz (~4 GB). Precisa de internet e leva de
echo  10 a 30 minutos na primeira vez.
echo.

REM -- 1. Placa NVIDIA -------------------------------------------
echo [1/5] Verificando placa de video NVIDIA...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo      ERRO: nao encontrei uma placa NVIDIA ^(nvidia-smi^).
    echo      O LuminaDub precisa de uma placa NVIDIA com driver atualizado.
    pause
    exit /b 1
)
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name^,memory.total --format^=csv^,noheader') do echo      %%g

REM -- 2. Python 3.11 --------------------------------------------
echo.
echo [2/5] Verificando Python 3.11...
set "PYBASE="
py -3.11 --version >nul 2>&1 && set "PYBASE=py -3.11"
if not defined PYBASE (
    echo      Python 3.11 nao encontrado.
    echo      Instale com:  winget install Python.Python.3.11
    echo      ^(ou baixe em python.org^) e rode este instalar.bat de novo.
    pause
    exit /b 1
)
%PYBASE% --version

REM -- 3. FFmpeg -------------------------------------------------
echo.
echo [3/5] Verificando FFmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo      FFmpeg nao encontrado. Ele e usado para extrair e montar o audio do video.
    echo      Instale com:  winget install Gyan.FFmpeg
    echo      Depois FECHE e abra de novo esta janela e rode o instalar.bat outra vez.
    pause
    exit /b 1
)
echo      OK.

REM -- 4. Ambiente Python + bibliotecas --------------------------
echo.
echo [4/5] Criando ambiente Python e instalando bibliotecas...
if not exist ".venv-fast\Scripts\python.exe" (
    %PYBASE% -m venv .venv-fast || goto :erro
)
set "PYEXE=%~dp0.venv-fast\Scripts\python.exe"
"%PYEXE%" -m pip install --upgrade pip -q || goto :erro
echo      PyTorch com CUDA ^(download grande, ~2.5 GB^)...
"%PYEXE%" -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 || goto :erro
echo      Demais bibliotecas...
"%PYEXE%" -m pip install -r requirements.txt || goto :erro

REM -- 5. Modelo de voz ------------------------------------------
echo.
echo [5/5] Baixando o modelo de voz Qwen3-TTS 1.7B ^(~4 GB^)...
"%PYEXE%" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='data/models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base')" || goto :erro
REM Whisper, Demucs e o tradutor sao baixados sozinhos no primeiro video.

echo.
echo  +======================================+
echo  ^|   Pronto! Agora e so abrir start.bat  ^|
echo  +======================================+
echo.
pause
exit /b 0

:erro
echo.
echo  ERRO na instalacao. Veja a mensagem acima.
echo  Se for problema de internet, e so rodar o instalar.bat de novo:
echo  ele continua de onde parou.
pause
exit /b 1
