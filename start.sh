#!/bin/bash

# RAG PDF Chat - Startup Script

echo "🤖 RAG PDF Chat Application"
echo "=========================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key!"
    echo "   Or use the local version with: python app_local.py"
    echo ""
fi

echo ""
echo "Choose your preferred mode:"
echo "1. OpenAI API (requires API key) - Better results"
echo "2. Local models (free) - No API key needed"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Starting application with OpenAI API..."
        python app.py
        ;;
    2)
        echo "Starting application with local models..."
        python app_local.py
        ;;
    *)
        echo "Invalid choice. Starting with local models by default..."
        python app_local.py
        ;;
esac
