@echo off
setlocal

:: 1. 이 배치 파일이 있는 폴더(프로젝트 루트)로 이동
cd /d "%~dp0"

echo ==========================================
echo   Vedic AI Integrated Launcher
echo   Location: %CD%
echo ==========================================
echo.

:: 2. Backend 실행 (FastAPI)
:: 새 창을 열고 backend 폴더로 들어간 뒤 python main.py 실행
echo [INFO] Launching Backend Server...
if exist "backend\main.py" (
    start "Vedic AI Backend" cmd /k "cd backend && python main.py"
) else (
    echo [ERROR] backend/main.py not found!
    pause
    exit /b 1
)

:: 백엔드가 포트를 점유할 시간을 잠시 줌 (2초)
timeout /t 2 /nobreak >nul

:: 3. Frontend 실행 (Next.js)
:: 새 창을 열고 frontend 폴더로 들어간 뒤 npm run dev 실행
echo [INFO] Launching Frontend Server...
if exist "frontend\package.json" (
    start "Vedic AI Frontend" cmd /k "cd frontend && npm run dev"
) else (
    echo [ERROR] frontend/package.json not found!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] All services started in new windows.
echo.
pause