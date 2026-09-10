@echo off
title DocuMind AI Launcher
echo ========================================================
echo               Launching DocuMind AI
echo ========================================================
echo.

if not exist venv\Scripts\streamlit.exe (
    echo [ERROR] Virtual environment not found. Please set up the environment first.
    pause
    exit /b 1
)

echo Activating virtual environment and starting Streamlit...
.\venv\Scripts\streamlit.exe run app.py
pause
