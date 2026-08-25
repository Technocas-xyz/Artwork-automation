@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting server at http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
