@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup...
    python "%SCRIPT_DIR%setup.py"
)

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%ocr.py" %*
