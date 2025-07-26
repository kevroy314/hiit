#!/bin/bash
# HIIT Analyzer Server Startup Script

echo "🏃 Starting HIIT Analyzer..."

# Create required directories if they don't exist
echo "📁 Checking directories..."
mkdir -p data settings

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "🐍 Python version: $PYTHON_VERSION"

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found"
    exit 1
fi

# Start the application
echo "🚀 Starting application..."
echo "📍 Access the application at http://localhost:8050"
python3 -m src.app