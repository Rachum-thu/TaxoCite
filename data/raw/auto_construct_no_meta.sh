#!/bin/bash

# Main pipeline script for automatic construction of paper dataset
# Usage: auto_construct.sh <directory_path>
# Example: auto_construct.sh data/raw/cs/retrieval_augmented_generation

# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 data/raw/cs/retrieval_augmented_generation"
    exit 1
fi

DIR="$1"

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

echo "=========================================="
echo "Processing directory: $DIR"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run each pipeline step
# bash "$SCRIPT_DIR/util/convert_pdf_to_txt.sh" "$DIR"
# bash "$SCRIPT_DIR/util/convert_txt_to_md.sh" "$DIR"
# bash "$SCRIPT_DIR/util/generate_taxonomy.sh" "$DIR"
# bash "$SCRIPT_DIR/util/segment_papers.sh" "$DIR"
bash "$SCRIPT_DIR/util/locate_references.sh" "$DIR" --mode no_meta

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
