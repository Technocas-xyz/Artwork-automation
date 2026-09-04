@echo off
REM ===========================================================================
REM  Build ArtworkAgent.exe (one-file) with ONLY Chromium bundled.
REM  Run on a Windows machine inside the project's venv.
REM  Output: dist\ArtworkAgent.exe  (then copy it into the server's downloads\)
REM ===========================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Could not find .venv - create it and install requirements first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"

echo === Ensuring build tools and Chromium are present ===
python -m pip install --quiet pyinstaller pystray requests python-dotenv
python -m playwright install chromium

echo === Building (only Chromium bundled; server-only modules excluded) ===
python build_agent.py

endlocal
