@echo off
rem Live webcam emotion recognition (NOT streamlit - uses your real webcam).
rem Uses Python 3.12 (has the DeepFace/TensorFlow emotion model installed).
rem Double-click to start. Press q or Esc in the video window to quit.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    echo Starting LIVE WEBCAM emotion recognition with Python 3.12...
    py -3.12 realtime_webcam.py --camera 1
    if %errorlevel% neq 0 (
        echo.
        echo Camera index 1 failed (privacy settings or wrong index?). Trying index 0...
        py -3.12 realtime_webcam.py --camera 0
    )
) else (
    echo 'py' launcher not found, trying default python...
    python realtime_webcam.py --camera 1
)
pause