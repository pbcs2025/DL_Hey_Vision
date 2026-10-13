@echo off
title AI Vision Assistant
cd /d "%~dp0"

echo ========================================================
echo        AI VISION ASSISTANT - STARTING APPLICATION
echo ========================================================
echo.
echo [1/3] Checking environment & server port 8000...

netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Backend server is already running on port 8000.
) else (
    echo Starting FastAPI & Vision Engine Server...
    start /min "Vision Server" python -m uvicorn ui.backend.app_server:app --host 0.0.0.0 --port 8000
    echo Waiting for AI models to initialize (5s)...
    timeout /t 5 /nobreak >nul
)

for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set LOCAL_IP=%%a
    goto :found_ip
)
:found_ip
set LOCAL_IP=%LOCAL_IP: =%
echo.
echo ========================================================
echo  Access URL (This Computer): http://localhost:8000
echo  Mobile / Phone URL (Wi-Fi): http://%LOCAL_IP%:8000
echo ========================================================

echo.
echo [2/3] Launching AI Vision Assistant Desktop Window...

if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:8000 --window-size=1280,840
) else if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
    start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:8000 --window-size=1280,840
) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://127.0.0.1:8000 --window-size=1280,840
) else (
    start http://127.0.0.1:8000
)

echo.
echo [3/3] Application running! Enjoy your session.
exit
