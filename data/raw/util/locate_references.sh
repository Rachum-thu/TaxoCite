#!/bin/bash

# Locate references in segmented papers
# Usage: locate_references.sh <directory> [--mode <mode>]
# Default mode: acm

# Parse arguments
DIR=""
MODE="acm"  # Default mode

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            if [ -z "$DIR" ]; then
                DIR="$1"
            fi
            shift
            ;;
    esac
done

echo "=== Step 6: Reference Location ==="

# Check if seg_papers directory exists
if [ ! -d "$DIR/seg_papers" ]; then
    echo "Warning: seg_papers directory not found, skipping reference location."
    exit 0
fi

# For no_meta mode, create yaml_refs directory if it doesn't exist
if [ "$MODE" = "no_meta" ]; then
    mkdir -p "$DIR/yaml_refs"
fi

# Check if yaml_refs directory exists (for non-no_meta modes)
if [ "$MODE" != "no_meta" ] && [ ! -d "$DIR/yaml_refs" ]; then
    echo "Warning: yaml_refs directory not found, skipping reference location."
    exit 0
fi

seg_files=()
ref_files=()

for seg_file in "$DIR/seg_papers"/*.yaml; do
    [ -f "$seg_file" ] || continue
    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"

    # For no_meta mode, don't check if ref_file exists (we'll create it)
    if [ "$MODE" = "no_meta" ]; then
        seg_files+=("$seg_file")
        ref_files+=("$ref_file")
    else
        # For other modes, only process if ref_file exists
        if [ -f "$ref_file" ]; then
            seg_files+=("$seg_file")
            ref_files+=("$ref_file")
        fi
    fi
done

if [ ${#seg_files[@]} -gt 0 ]; then
    echo "Locating references in ${#seg_files[@]} papers (mode: $MODE)..."
    python -m data.raw.util.locate_ref \
        --seg_papers "${seg_files[@]}" \
        --yaml_refs "${ref_files[@]}" \
        --mode "$MODE"
    echo "Reference location completed!"
else
    echo "No seg_papers found to process."
fi
