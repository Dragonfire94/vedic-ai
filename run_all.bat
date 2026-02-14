@echo off
setlocal

cd /d "%~dp0"

echo ==========================================
echo   Vedic AI Integrated Launcher
echo   Location: %CD%
echo ==========================================
echo.

if not exist "backend\main.py" (
    echo [ERROR] backend\main.py not found!
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo [ERROR] frontend\package.json not found!
    pause
    exit /b 1
)

echo [INFO] Running backend dependency preflight...
python -c "import fastapi, swisseph, timezonefinder" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Missing backend dependencies detected. Installing from backend\requirements.txt...
    python -m pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )

    python -c "import fastapi, swisseph, timezonefinder" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Dependencies still missing after installation.
        echo [HINT] Run: python -m pip install -r backend\requirements.txt
        pause
        exit /b 1
    )
)

echo [INFO] Launching Backend Server...
start "Vedic AI Backend" cmd /k "python -m backend.main"

timeout /t 2 /nobreak >nul

echo [INFO] Launching Frontend Server...
start "Vedic AI Frontend" cmd /k "cd frontend && npm.cmd run dev"

echo.
echo [SUCCESS] All services started in new windows.
echo.
pause
