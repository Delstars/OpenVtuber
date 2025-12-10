@echo off
TITLE OpenVtuber Launcher
echo Checking system requirements...

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is NOT installed!
    echo Please install Python from python.org and tick "Add to PATH" during installation.
    PAUSE
    EXIT
)

:: Check/Install Requirements automatically
if not exist "tracker\.installed" (
    echo First run detected. Installing libraries...
    pip install -r tracker/requirements.txt
    echo done > tracker\.installed
)

echo Starting OpenVtuber Tracker...
cd tracker
python tracker_gui.py