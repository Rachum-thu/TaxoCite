#!/bin/bash

# Extract references from TXT papers
# Usage: extract_references.sh <directory>

DIR="$1"

echo "=== Step 3: Reference Extraction ==="

# Create yaml_refs directory
mkdir -p "$DIR/yaml_refs"

# Extract references from txt_papers
ref_input_files=()
ref_output_files=()

for txt_file in "$DIR/txt_papers"/*.txt; do
    [ -f "$txt_file" ] || continue
    filename=$(basename "$txt_file" .txt)
    ref_input_files+=("$txt_file")
    ref_output_files+=("$DIR/yaml_refs/${filename}.yaml")
done

# Batch extract references
if [ ${#ref_input_files[@]} -gt 0 ]; then
    echo "Extracting references from ${#ref_input_files[@]} txt_papers files..."
    python -m data.raw.util.txt2ref --inputs "${ref_input_files[@]}" --outputs "${ref_output_files[@]}"
    echo "Reference extraction completed!"
else
    echo "No txt files found in txt_papers for reference extraction."
fi
