#!/bin/bash

# Locate references in segmented papers
# Usage: locate_references.sh <directory>

DIR="$1"

echo "=== Step 6: Reference Location ==="

if [ -d "$DIR/seg_papers" ] && [ -d "$DIR/yaml_refs" ]; then
    seg_files=()
    ref_files=()

    for seg_file in "$DIR/seg_papers"/*.yaml; do
        [ -f "$seg_file" ] || continue
        filename=$(basename "$seg_file" .yaml)
        ref_file="$DIR/yaml_refs/${filename}.yaml"

        if [ -f "$ref_file" ]; then
            seg_files+=("$seg_file")
            ref_files+=("$ref_file")
        fi
    done

    if [ ${#seg_files[@]} -gt 0 ]; then
        echo "Locating references in ${#seg_files[@]} papers..."
        python -m data.raw.util.locate_ref \
            --seg_papers "${seg_files[@]}" \
            --yaml_refs "${ref_files[@]}" \
            --mode acm
        echo "Reference location completed!"
    else
        echo "No matching seg_papers and yaml_refs found."
    fi
else
    echo "Warning: seg_papers or yaml_refs directory not found, skipping reference location."
fi
