@echo off
REM Script to set up a virtual environment and run tests on Windows

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run tests
echo Running Phase 1 tests...
python tests\test_phase1.py

REM Deactivate virtual environment
deactivate
