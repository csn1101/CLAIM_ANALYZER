@echo off
echo ========================================
echo Insurance Claim Analyzer Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Running test demonstration...
python test_demo.py
if errorlevel 1 (
    echo ERROR: Test failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting the web application...
echo ========================================
echo.
echo The application will be available at:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
python app.py
