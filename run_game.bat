@echo off
rem AI Face Emotion Game launcher
rem Uses Python 3.12 (has the DeepFace/TensorFlow emotion model installed).
rem Double-click this file to start the game at http://localhost:8501
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    echo Starting AI Face Emotion Game with Python 3.12...
    py -3.12 -m streamlit run app.py
    if %errorlevel% neq 0 (
        echo.
        echo Python 3.12 may not be installed. Trying default python...
        python -m streamlit run app.py
    )
) else (
    echo 'py' launcher not found, trying default python...
    python -m streamlit run app.py
)

pause