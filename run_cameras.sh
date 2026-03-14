#!/bin/bash

# A script to launch the entry and exit cameras simultaneously
cd /home/katomaran/project/PROJECT_LPR_CRAFTSMAN/lpr_development

echo "Starting Entry Camera Process..."
python camera.py --camera_name entry_camera --config_file config.json &
ENTRY_PID=$!

echo "Starting Exit Camera Process..."
python camera.py --camera_name exit_camera --config_file config.json &
EXIT_PID=$!

echo "Both cameras are now running in the background."
echo "Entry Camera PID: $ENTRY_PID"
echo "Exit Camera PID: $EXIT_PID"
echo ""
echo "Press [CTRL+C] to stop both cameras."

# Wait for background processes to finish (or until user stops them)
wait $ENTRY_PID
wait $EXIT_PID
