@echo off
setlocal

if /i not "%~1"=="--background" (
  powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "Start-Process -FilePath '%~f0' -ArgumentList '--background' -WindowStyle Hidden"
  exit /b 0
)

cd /d "%~dp0"

set "ONEBOT_WS_MODE=reverse"
set "ONEBOT_WS_URL=ws://127.0.0.1:18001/ws"
set "WEB_ENABLED=true"
set "WEB_HOST=127.0.0.1"
set "WEB_PORT=18002"
set "SKIP_MANAGED_NAPCAT=0"

set "BOT_QQ="
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /r /c:"^[ ]*BOT_QQ[ ]*=" ".env"`) do (
    set "BOT_QQ=%%B"
  )
)
set "BOT_QQ=%BOT_QQ:"=%"
set "BOT_QQ=%BOT_QQ: =%"

set "PYTHON_EXE=python"
if exist ".\venv\Scripts\python.exe" (
  set "PYTHON_EXE=%cd%\venv\Scripts\python.exe"
)
if exist ".\.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%cd%\.venv\Scripts\python.exe"
)

for %%P in (NapCat.exe NapCatWinBootMain.exe) do (
  taskkill /f /im %%P /t >nul 2>&1
)
for %%I in (18001 18002) do (
  for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%%I .*LISTENING"') do (
    taskkill /f /pid %%P >nul 2>&1
  )
)

start "" /b "%PYTHON_EXE%" main.py

timeout /t 5 /nobreak >nul
start "" "http://localhost:18002"
exit /b 0
