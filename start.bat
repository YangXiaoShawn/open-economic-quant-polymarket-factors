@echo off
echo Polymarket Arb Scanner — Starting...
echo.

cd /d "%~dp0"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
