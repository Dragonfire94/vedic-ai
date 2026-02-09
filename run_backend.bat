@echo off
setlocal

cd /d C:\dev\vedic-ai

if not exist venv\Scripts\python.exe (
  echo [ERROR] venv not found: C:\dev\vedic-ai\venv
  pause
  exit /b 1
)

echo [INFO] Starting backend (FastAPI)...
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

pause
