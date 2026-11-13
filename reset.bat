@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title LuminaDub AI - Reset Total
echo.
echo  +======================================+
echo  ^|     LuminaDub AI - RESET TOTAL        ^|
echo  +======================================+
echo.
echo  Isso vai APAGAR PARA SEMPRE:
echo    - Todos os videos enviados (data\projects\*\input)
echo    - Todo o progresso e videos dublados (work\ e output\)
echo    - Projetos pausados (data\paused_projects)
echo.
echo  NAO vai apagar:
echo    - Os modelos de IA ja baixados (data\models) - ficam prontos pra uso
echo    - Nada fora da pasta do site (seus downloads ja copiados estao seguros)
echo.
echo  Se voce ja baixou os videos que queria, pode seguir tranquilo.
echo.
set /p CONFIRM="Digite SIM para confirmar o reset: "
if /i not "%CONFIRM%"=="SIM" (
    echo.
    echo  Cancelado. Nada foi apagado.
    pause
    exit /b 0
)

echo.
echo [1/3] Parando o servidor, se estiver rodando...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    echo      Matando PID %%a na porta 5000...
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":7860 " ^| findstr "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo      OK.

echo.
echo [2/3] Apagando projetos...
if exist "data\projects" rmdir /s /q "data\projects" 2>nul
if exist "data\paused_projects" rmdir /s /q "data\paused_projects" 2>nul
mkdir "data\projects"
echo      Projetos apagados. Pasta data\projects recriada vazia.

echo.
echo [3/3] Limpando cache Python...
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
if exist "stages\__pycache__" rmdir /s /q "stages\__pycache__" 2>nul
if exist "models\__pycache__" rmdir /s /q "models\__pycache__" 2>nul
if exist "ui\__pycache__" rmdir /s /q "ui\__pycache__" 2>nul
echo      OK.

echo.
echo  +======================================+
echo  ^|   Site zerado. Pronto pra usar!      ^|
echo  ^|   Rode start.bat para iniciar.       ^|
echo  +======================================+
echo.
pause
