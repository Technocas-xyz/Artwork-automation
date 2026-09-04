@echo off
REM ===========================================================================
REM  Artwork Studio - designer agent launcher
REM  Double-click this file to start the agent on your PC.
REM ===========================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo.
  echo   Could not find the virtual environment (.venv).
  echo   Follow SETUP.md first - you only need to do that once.
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting the artwork agent... leave this window open while you work.
echo Press Ctrl+C to stop.
echo.

python agent.py

echo.
echo The agent has stopped. Close this window, or run it again to reconnect.
pause
