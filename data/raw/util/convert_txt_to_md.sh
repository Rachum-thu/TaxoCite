#!/bin/bash

# Convert TXT files to Markdown
# Usage: convert_txt_to_md.sh <directory>

DIR="$1"

echo "=== Step 2: TXT to MD Conversion ==="

# Create markdown output directories
mkdir -p "$DIR/md_papers"
mkdir -p "$DIR/md_surveys"

# Collect all txt files and their output paths
input_files=()
output_files=()

# Add txt_papers files
for txt_file in "$DIR/txt_papers"/*.txt; do
    [ -f "$txt_file" ] || continue
    filename=$(basename "$txt_file" .txt)
    input_files+=("$txt_file")
    output_files+=("$DIR/md_papers/${filename}.md")
done

# Add txt_surveys files
for txt_file in "$DIR/txt_surveys"/*.txt; do
    [ -f "$txt_file" ] || continue
    filename=$(basename "$txt_file" .txt)
    input_files+=("$txt_file")
    output_files+=("$DIR/md_surveys/${filename}.md")
done

# Batch convert all at once
if [ ${#input_files[@]} -gt 0 ]; then
    echo "Converting ${#input_files[@]} txt files to markdown..."
    python -m data.raw.util.txt2md --inputs "${input_files[@]}" --outputs "${output_files[@]}"
    echo "TXT to MD conversion completed!"
else
    echo "No txt files found to convert."
fi
