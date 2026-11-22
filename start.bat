@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title LuminaDub AI - Iniciando...
echo.
echo  +======================================+
echo  ^|     LuminaDub AI - Limpeza e Inicio  ^|
echo  +======================================+
echo.

REM .venv-fast = PyTorch 2.7 + transformers 5 + faster-qwen3-tts (voz ~2x mais rapida).
REM O ambiente antigo continua em .venv como reserva.
set "PYEXE=%~dp0.venv-fast\Scripts\python.exe"
REM Perfil de hardware (config.py -> PROFILES): ajustado pra RTX 3060 12GB
set "DUB_PROFILE=rtx3060"
if not exist "%PYEXE%" (
    echo      ERRO: venv nao encontrado em .venv-fast\Scripts\python.exe
    echo      Rode: py -3.11 -m venv .venv-fast  e depois instale requirements.txt
    pause
    exit /b 1
)

echo [1/5] Limpando cache Python...
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
if exist "stages\__pycache__" rmdir /s /q "stages\__pycache__" 2>nul
if exist "models\__pycache__" rmdir /s /q "models\__pycache__" 2>nul
if exist "ui\__pycache__" rmdir /s /q "ui\__pycache__" 2>nul
echo      Cache removido.

REM Projetos antigos (data\projects\*) so sao apagados se voce pedir (passo de
REM limpeza abaixo): por padrao o servidor retoma sozinho, no boot, qualquer
REM video que ficou pela metade.

echo.
echo [2/5] Verificando dependencias...
"%PYEXE%" -m pip install waitress >nul 2>&1
echo      OK.

echo.
echo [3/5] Matando processos nas portas 5000 e 7860...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    echo      Matando PID %%a na porta 5000...
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    echo      Matando PID %%a na porta 7860...
    taskkill /f /pid %%a >nul 2>&1
)
echo      Portas liberadas.

echo.
echo  Limpar TUDO? Apaga todos os projetos, a fila e os videos dublados.
echo  (Os videos originais dos cursos nao sao tocados. Sem resposta = Nao.)
choice /C SN /T 8 /D N /M "  Limpar tudo"
if errorlevel 2 (
    echo      Mantendo projetos e videos.
) else (
    REM Can run right after the old server was killed: give its files a moment to unlock.
    timeout /t 2 /nobreak >nul
    if exist "data\projects" rmdir /s /q "data\projects" 2>nul
    if exist "data\projects" (
        echo      AVISO: alguns arquivos estavam em uso e nao foram apagados.
    ) else (
        echo      Tudo limpo. O site vai abrir vazio.
    )
    if not exist "data\projects" mkdir "data\projects"
)

echo.
echo [4/5] Verificando Ollama...
where ollama >nul 2>&1
if !errorlevel! equ 0 (
    echo      Ollama encontrado.
) else (
    echo      AVISO: Ollama nao encontrado. Traducao IA nao estara disponivel.
    echo      Instale em: https://ollama.com
)

echo.
echo [5/5] Iniciando LuminaDub AI...
echo      Servidor: http://localhost:5000
echo      (Se o servidor cair, ele reinicia sozinho automaticamente.
echo       Feche esta janela pra parar de vez. CTRL+C nao basta - segure
echo       ou feche a janela.)
echo.

set "RESTART_COUNT=0"

:serverloop
"%PYEXE%" server.py
set "EXIT_CODE=%errorlevel%"

if "%EXIT_CODE%"=="0" (
    echo.
    echo  Servidor encerrado normalmente ^(codigo 0^). Nao reiniciando.
    goto end
)

set /a RESTART_COUNT+=1
echo.
echo  +======================================+
echo  ^| Servidor caiu ^(codigo %EXIT_CODE%^). Reinicio #%RESTART_COUNT% ^|
echo  +======================================+
echo  Reiniciando em 5 segundos... ^(feche esta janela pra cancelar^)
timeout /t 5 /nobreak >nul
goto serverloop

:end
pause
