#!/bin/bash

# HIIT Analyzer - Dash Version Launcher
echo "🏃‍♂️ Starting HIIT Analyzer (Dash Version)..."

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir -p data
    echo "⚠️  Please add your .fit files to the 'data/' directory"
fi

# Check if settings directory exists
if [ ! -d "settings" ]; then
    echo "📁 Creating settings directory..."
    mkdir -p settings
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Check for requirements
if [ ! -f "requirements_dash.txt" ]; then
    echo "❌ requirements_dash.txt not found"
    exit 1
fi

# Try to run with virtual environment first
if [ -d "venv" ]; then
    echo "🐍 Using existing virtual environment..."
    source venv/bin/activate
    python app_dash.py
elif command -v docker &> /dev/null; then
    echo "🐳 Python venv not available, trying Docker..."
    docker-compose up --build hiit-analyzer-dash
else
    echo "⚠️  Virtual environment not found. Trying system Python..."
    echo "💡 Consider using Docker for easier deployment: docker-compose up --build"
    python3 app_dash.py
fi