@echo off
echo 🏃‍♂️ Starting HIIT Analyzer (Dash Version)...

REM Check if data directory exists
if not exist "data" (
    echo 📁 Creating data directory...
    mkdir data
    echo ⚠️  Please add your .fit files to the 'data/' directory
)

REM Check if settings directory exists
if not exist "settings" (
    echo 📁 Creating settings directory...
    mkdir settings
)

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check for requirements
if not exist "requirements_dash.txt" (
    echo ❌ requirements_dash.txt not found
    pause
    exit /b 1
)

REM Try to run the application
echo 🚀 Starting Dash application...
python app_dash.py

pause