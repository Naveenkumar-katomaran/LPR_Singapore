#!/bin/bash

# A script to launch the entry and exit cameras simultaneously
cd /home/katomaran/project/PROJECT_LPR_CRAFTSMAN/lpr_development

echo "Starting Entry Camera Process..."
python camera.py --camera_name entry_camera --config_file config.json &
ENTRY_PID=$!

echo "Starting Exit Camera Process..."
python camera.py --camera_name exit_camera --config_file config.json &
EXIT_PID=$!

# Function to kill child processes when the script is stopped
cleanup() {
    echo ""
    echo "Stopping cameras immediately..."
    kill -9 $ENTRY_PID 2>/dev/null
    kill -9 $EXIT_PID 2>/dev/null
    exit 0
}

# Trap CTRL+C (SIGINT), CTRL+Z (SIGTSTP), and termination
trap cleanup SIGINT SIGTSTP SIGTERM

echo "Both cameras are now running in the background."
echo "Entry Camera PID: $ENTRY_PID"
echo "Exit Camera PID: $EXIT_PID"
echo ""
echo "Press [CTRL+C] or [CTRL+Z] to stop both cameras."

# Wait for background processes to finish
wait $ENTRY_PID
wait $EXIT_PID

