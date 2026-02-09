@echo off
setlocal

cd /d C:\dev\vedic-ai

if not exist venv\Scripts\python.exe (
  echo [ERROR] venv not found: C:\dev\vedic-ai\venv
  pause
  exit /b 1
)

echo [INFO] Launching backend in a new window...
start "Vedic AI Backend" cmd /k "cd /d C:\dev\vedic-ai && venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

timeout /t 2 >nul

echo [INFO] Launching frontend in a new window...
start "Vedic AI Frontend" cmd /k "cd /d C:\dev\vedic-ai && venv\Scripts\python.exe -m streamlit run frontend\app.py"

echo [OK] Both launched.
echo - Backend:  http://127.0.0.1:8000/health
echo - Frontend: Streamlit will open in browser automatically
pause
