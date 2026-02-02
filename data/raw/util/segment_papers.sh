#!/bin/bash

# Segment markdown papers into blocks
# Usage: segment_papers.sh <directory>

DIR="$1"

echo "=== Step 5: Paper Segmentation ==="

# Create seg_papers directory
mkdir -p "$DIR/seg_papers"

if [ -d "$DIR/md_papers" ]; then
    input_files=()
    output_files=()

    for md_file in "$DIR/md_papers"/*.md; do
        [ -f "$md_file" ] || continue
        filename=$(basename "$md_file" .md)
        input_files+=("$md_file")
        output_files+=("$DIR/seg_papers/${filename}.yaml")
    done

    if [ ${#input_files[@]} -gt 0 ]; then
        echo "Segmenting ${#input_files[@]} markdown papers..."
        python -m data.raw.util.md2seg --inputs "${input_files[@]}" --outputs "${output_files[@]}"
        echo "Segmentation completed!"
    else
        echo "No markdown papers found to segment."
    fi
else
    echo "Warning: md_papers directory not found, skipping segmentation."
fi
