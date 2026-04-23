#!/usr/bin/env bash
set -euo pipefail

TARGET_PANE="%100"
MESSAGE="This is an automated ping to make sure everything is going well ! 1. Check the state of the runs, if there are any bugs, fix them 2. Check the instructions, if any tests haven't been implemented and run, do it 3. If you're still receiving these and we're all done, it means there's extra time. First double-check previous results. Then, look at all of the results, write down a results.md about your findings, and double down on the most promising directions, but keep the code separate from the previous campaign code so we can version control and reproduce results. Launch another campaign about as big as this one is scope and runtime, and keep iterating."

# Clear any existing draft in the input line before injecting the ping.
tmux send-keys -t "$TARGET_PANE" Escape C-u

# Use tmux paste-buffer for better TUI compatibility, then submit explicitly.
tmux set-buffer -- "$MESSAGE"
tmux paste-buffer -t "$TARGET_PANE" -d

# Give the target app a moment to process the paste before submit.
sleep 0.35
tmux send-keys -t "$TARGET_PANE" Enter
