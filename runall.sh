#!/bin/bash

# Directory path
DIR="/home/noman/Documents/Abdulah/consforc web models"

# Change to the directory
cd "$DIR"

# Activate virtual environment
source venv/bin/activate

echo "Starting all services with uvicorn..."

# Start chat_sentiment_app on port 8001
echo "Starting chat_sentiment_app on port 8001..."
uvicorn chat_sentiment_app:app --host 0.0.0.0 --port 8001 &
CHAT_PID=$!

# Start image_app on port 8002
echo "Starting image_app on port 8002..."
uvicorn image_app:app --host 0.0.0.0 --port 8002 &
IMAGE_PID=$!

# Start object_posture_app on port 8003
echo "Starting object_posture_app on port 8003..."
uvicorn object_posture_app:app --host 0.0.0.0 --port 8003 &
OBJECT_PID=$!

# Start translation_audio_app on port 8004
echo "Starting translation_audio_app on port 8004..."
uvicorn translation_audio_app:app --host 0.0.0.0 --port 8004 &
AUDIO_PID=$!

echo "All services started!"
echo "Chat & Sentiment: http://localhost:8001"
echo "Image Generation: http://localhost:8002"
echo "Object Detection & Posture: http://localhost:8003"
echo "Translation & Audio: http://localhost:8004"

# Function to kill all processes
cleanup() {
    echo "Stopping all services..."
    kill $CHAT_PID $IMAGE_PID $OBJECT_PID $AUDIO_PID 2>/dev/null
    echo "All services stopped."
    exit 0
}

# Trap Ctrl+C to cleanup
trap cleanup SIGINT

# Wait for all background processes
wait