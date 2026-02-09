@echo off
setlocal

cd /d C:\dev\vedic-ai

if not exist venv\Scripts\python.exe (
  echo [ERROR] venv not found: C:\dev\vedic-ai\venv
  pause
  exit /b 1
)

echo [INFO] Starting frontend (Streamlit)...
venv\Scripts\python.exe -m streamlit run frontend\app.py

pause
