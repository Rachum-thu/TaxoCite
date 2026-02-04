#!/bin/bash

# Script to run all experiments sequentially
# Exit on error
set -e

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run command with logging
run_command() {
    local cmd="$1"
    local step="$2"
    local total="$3"

    echo -e "${BLUE}[${step}/${total}] Running: ${cmd}${NC}"
    if eval "$cmd"; then
        echo -e "${GREEN}[${step}/${total}] Success!${NC}\n"
    else
        echo -e "${RED}[${step}/${total}] Failed!${NC}"
        exit 1
    fi
}

# Total number of commands
TOTAL=5
STEP=1

echo "Starting all experiments..."
echo "================================"

# gpt-5-mini-2025-08-07 experiments
run_command "bash method/run_experiments.sh data/raw/gis/climate gpt-5-mini-2025-08-07" $STEP $TOTAL
STEP=$((STEP + 1))

run_command "bash method/run_experiments.sh data/raw/cs/generative_retrieval gpt-5-mini-2025-08-07" $STEP $TOTAL
STEP=$((STEP + 1))

run_command "bash method/run_experiments.sh data/raw/gis/regionalization gpt-5-mini-2025-08-07" $STEP $TOTAL
STEP=$((STEP + 1))

run_command "bash method/run_experiments.sh data/raw/cs/retrieval_augmented_generation gpt-5-mini-2025-08-07" $STEP $TOTAL
STEP=$((STEP + 1))

run_command "bash method/run_experiments.sh data/raw/chem/aggregation-induced_emission gpt-5-mini-2025-08-07" $STEP $TOTAL

echo "================================"
echo -e "${GREEN}All experiments completed successfully!${NC}"
