@echo off
chcp 65001 >nul
cd /d "%~dp0"
title LuminaDub AI - Baixar Videos Dublados

set "DEST=%USERPROFILE%\Downloads"
set /a COPIADOS=0
set /a PULADOS=0

echo.
echo  +======================================+
echo  ^|   LuminaDub AI - Baixar Videos        ^|
echo  +======================================+
echo.
echo  Origem : %~dp0data\projects\*\output
echo  Destino: %DEST%
echo.

if not exist "data\projects" (
    echo  Nenhum projeto encontrado em data\projects.
    goto fim
)

for /d %%P in ("data\projects\*") do (
    for %%F in ("%%P\output\*_dublado_pt.mp4") do (
        if exist "%DEST%\%%~nxF" (
            set /a PULADOS+=1
        ) else (
            echo  Copiando: "%%~nxF"
            copy /y "%%F" "%DEST%\" >nul
            if errorlevel 1 (
                echo    ERRO ao copiar este arquivo.
            ) else (
                set /a COPIADOS+=1
            )
        )
    )
)

:fim
echo.
echo  +======================================+
echo   Copiados agora : %COPIADOS%
echo   Ja estavam la  : %PULADOS%
echo  +======================================+
echo.
echo  Abrindo a pasta Downloads...
start "" "%DEST%"
echo.
pause
