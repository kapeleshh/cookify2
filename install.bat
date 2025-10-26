@echo off
:: Cookify Installation Script for Windows

echo === Cookify Installation Script ===
echo This script will set up the Cookify environment and install dependencies.

:: Check if Python 3.8+ is installed
echo Checking Python version...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found.
    echo Please install Python 3.8+ and try again.
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%i in ('python --version') do set python_version=%%i
echo Found Python %python_version%

:: Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%python_version%") do (
    set major=%%a
    set minor=%%b
)

if %major% LSS 3 (
    echo Error: Cookify requires Python 3.8 or higher.
    echo Please install Python 3.8+ and try again.
    exit /b 1
) else (
    if %major% EQU 3 (
        if %minor% LSS 8 (
            echo Error: Cookify requires Python 3.8 or higher.
            echo Please install Python 3.8+ and try again.
            exit /b 1
        )
    )
)

:: Check if FFmpeg is installed
echo Checking FFmpeg installation...
ffmpeg -version > nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: FFmpeg not found.
    echo FFmpeg is required for video processing.
    echo Please download and install FFmpeg from https://ffmpeg.org/download.html
    echo Make sure to add FFmpeg to your PATH environment variable.
) else (
    for /f "tokens=*" %%i in ('ffmpeg -version') do (
        echo Found %%i
        goto :ffmpeg_found
    )
    :ffmpeg_found
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install the package in development mode
echo Installing Cookify and dependencies...
pip install -e .

:: Download pre-trained models
echo Downloading pre-trained models...
python -m cookify.src.utils.model_downloader

echo.
echo === Installation Complete ===
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To use Cookify, run:
echo   cookify path\to\video.mp4
echo.

pause
