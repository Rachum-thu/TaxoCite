#!/bin/bash

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

echo "Processing directory: $DIR"

# Create output directories
mkdir -p "$DIR/txt_papers"
mkdir -p "$DIR/txt_surveys"

echo "Created output directories: txt_papers and txt_surveys"

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
    echo "All conversions completed!"
else
    echo "No txt files found to convert."
fi

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

# Generate topic taxonomy from surveys
TOPIC_NAME=$(basename "$DIR" | tr '_' ' ')

if [ -d "$DIR/md_surveys" ] && [ "$(ls -A "$DIR/md_surveys"/*.md 2>/dev/null)" ]; then
    echo "Generating taxonomy for topic: $TOPIC_NAME"
    python -m data.raw.util.surveys2taxo \
        --survey_dir "$DIR/md_surveys" \
        --topic_name "$TOPIC_NAME" \
        --output "$DIR/domain_topic.md"
    echo "Taxonomy generation completed!"
else
    echo "Warning: No survey markdown files found in $DIR/md_surveys, skipping taxonomy generation."
fi

# Segment markdown papers
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

# Locate references in segmented papers
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
