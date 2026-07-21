@echo off
REM Pokretac Loto Analizatora (web). Dupli klik za start.
cd /d "%~dp0"
title Loto Analizator - server

echo ============================================
echo   Loto Analizator - pokrecem server...
echo   Aplikacija ce se otvoriti u browseru.
echo   Za gasenje: zatvori ovaj prozor ili Ctrl+C
echo ============================================
echo.

REM Probaj 'python', pa 'py' ako prvi ne postoji
python pokreni.py
if errorlevel 1 (
    py pokreni.py
)

echo.
echo Server je zaustavljen. Mozes zatvoriti prozor.
pause
