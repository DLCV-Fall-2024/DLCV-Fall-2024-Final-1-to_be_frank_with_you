#!/bin/bash


name=$1
training_dir=$2
ckpt_path=$3

COMMAND="python -m src.inference --name $name --training_dir $training_dir --ckpt_path $ckpt_path"

TOTAL_SLICE=2  # Total number of dataset slices
CUDA_DEVICES=(0 1)  # List of CUDA devices to use

WORKING_DIR=$(pwd)
CONDA_ENV="dlcv-final"  # Replace with your Conda environment name

# Check if inside tmux
if [[ -z "$TMUX" ]]; then
    echo "Error: This script must be run inside a tmux session."
    exit 1
fi
# Get the current working directory
PWD=$(pwd)

panel_index=0

# Iterate over all panes and send the respective commands
for _pane in $(tmux list-panes -F '#P'); do
    # Send the 'cd' command to ensure each pane is in the correct directory
    tmux send-keys -t ${_pane} "cd $PWD" C-m
    
    # Activate the Conda environment
    tmux send-keys -t ${_pane} "conda activate $CONDA_ENV" C-m

    # Send the command with the appropriate CUDA device
    tmux send-keys -t ${_pane} "CUDA_VISIBLE_DEVICES=$panel_index $COMMAND --total_slices $TOTAL_SLICE --slice $panel_index" C-m
    panel_index=$((panel_index + 1))
done