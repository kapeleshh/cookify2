#!/bin/bash
# Script to set up a virtual environment and run tests

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run tests
echo "Running Phase 1 tests..."
python tests/test_phase1.py

# Deactivate virtual environment
deactivate
