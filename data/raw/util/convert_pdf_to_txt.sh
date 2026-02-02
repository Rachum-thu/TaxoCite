#!/bin/bash

# Convert PDFs to TXT files
# Usage: convert_pdf_to_txt.sh <directory>

DIR="$1"

echo "=== Step 1: PDF to TXT Conversion ==="

# Create output directories
mkdir -p "$DIR/txt_papers"
mkdir -p "$DIR/txt_surveys"

# Convert PDFs in pdf_papers to txt_papers
if [ -d "$DIR/pdf_papers" ]; then
    echo "Converting PDFs in pdf_papers..."
    for pdf_file in "$DIR/pdf_papers"/*.pdf; do
        if [ -f "$pdf_file" ]; then
            filename=$(basename "$pdf_file" .pdf)
            output_file="$DIR/txt_papers/${filename}.txt"
            echo "  Converting: $filename.pdf -> $filename.txt"
            python -m data.raw.util.pdf2txt "$pdf_file" "$output_file"
        fi
    done
else
    echo "Warning: $DIR/pdf_papers does not exist, skipping..."
fi

# Convert PDFs in pdf_surveys to txt_surveys
if [ -d "$DIR/pdf_surveys" ]; then
    echo "Converting PDFs in pdf_surveys..."
    for pdf_file in "$DIR/pdf_surveys"/*.pdf; do
        if [ -f "$pdf_file" ]; then
            filename=$(basename "$pdf_file" .pdf)
            output_file="$DIR/txt_surveys/${filename}.txt"
            echo "  Converting: $filename.pdf -> $filename.txt"
            python -m data.raw.util.pdf2txt "$pdf_file" "$output_file"
        fi
    done
else
    echo "Warning: $DIR/pdf_surveys does not exist, skipping..."
fi

echo "PDF to TXT conversion completed!"
