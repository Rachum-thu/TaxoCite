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

# # Create output directories
# mkdir -p "$DIR/txt_papers"
# mkdir -p "$DIR/txt_surveys"

# echo "Created output directories: txt_papers and txt_surveys"

# # Convert PDFs in pdf_papers to txt_papers
# if [ -d "$DIR/pdf_papers" ]; then
#     echo "Converting PDFs in pdf_papers..."
#     for pdf_file in "$DIR/pdf_papers"/*.pdf; do
#         if [ -f "$pdf_file" ]; then
#             filename=$(basename "$pdf_file" .pdf)
#             output_file="$DIR/txt_papers/${filename}.txt"
#             echo "  Converting: $filename.pdf -> $filename.txt"
#             python -m data.raw.util.pdf2txt "$pdf_file" "$output_file"
#         fi
#     done
# else
#     echo "Warning: $DIR/pdf_papers does not exist, skipping..."
# fi

# # Convert PDFs in pdf_surveys to txt_surveys
# if [ -d "$DIR/pdf_surveys" ]; then
#     echo "Converting PDFs in pdf_surveys..."
#     for pdf_file in "$DIR/pdf_surveys"/*.pdf; do
#         if [ -f "$pdf_file" ]; then
#             filename=$(basename "$pdf_file" .pdf)
#             output_file="$DIR/txt_surveys/${filename}.txt"
#             echo "  Converting: $filename.pdf -> $filename.txt"
#             python -m data.raw.util.pdf2txt "$pdf_file" "$output_file"
#         fi
#     done
# else
#     echo "Warning: $DIR/pdf_surveys does not exist, skipping..."
# fi

# echo "PDF to TXT conversion completed!"

# # Create markdown output directories
# mkdir -p "$DIR/md_papers"
# mkdir -p "$DIR/md_surveys"

# # Collect all txt files and their output paths
# input_files=()
# output_files=()

# # Add txt_papers files
# for txt_file in "$DIR/txt_papers"/*.txt; do
#     [ -f "$txt_file" ] || continue
#     filename=$(basename "$txt_file" .txt)
#     input_files+=("$txt_file")
#     output_files+=("$DIR/md_papers/${filename}.md")
# done

# # Add txt_surveys files
# for txt_file in "$DIR/txt_surveys"/*.txt; do
#     [ -f "$txt_file" ] || continue
#     filename=$(basename "$txt_file" .txt)
#     input_files+=("$txt_file")
#     output_files+=("$DIR/md_surveys/${filename}.md")
# done

# # Batch convert all at once
# if [ ${#input_files[@]} -gt 0 ]; then
#     echo "Converting ${#input_files[@]} txt files to markdown..."
#     python -m data.raw.util.txt2md --inputs "${input_files[@]}" --outputs "${output_files[@]}"
#     echo "All conversions completed!"
# else
#     echo "No txt files found to convert."
# fi

# # Create yaml_refs directory
# mkdir -p "$DIR/yaml_refs"

# # Extract references from txt_papers
# ref_input_files=()
# ref_output_files=()

# for txt_file in "$DIR/txt_papers"/*.txt; do
#     [ -f "$txt_file" ] || continue
#     filename=$(basename "$txt_file" .txt)
#     ref_input_files+=("$txt_file")
#     ref_output_files+=("$DIR/yaml_refs/${filename}.yaml")
# done

# # Batch extract references
# if [ ${#ref_input_files[@]} -gt 0 ]; then
#     echo "Extracting references from ${#ref_input_files[@]} txt_papers files..."
#     python -m data.raw.util.txt2ref --inputs "${ref_input_files[@]}" --outputs "${ref_output_files[@]}"
#     echo "Reference extraction completed!"
# else
#     echo "No txt files found in txt_papers for reference extraction."
# fi

# # Generate topic taxonomy from surveys
# TOPIC_NAME=$(basename "$DIR" | tr '_' ' ')

# if [ -d "$DIR/md_surveys" ] && [ "$(ls -A "$DIR/md_surveys"/*.md 2>/dev/null)" ]; then
#     echo "Generating taxonomy for topic: $TOPIC_NAME"
#     python -m data.raw.util.surveys2taxo \
#         --survey_dir "$DIR/md_surveys" \
#         --topic_name "$TOPIC_NAME" \
#         --output "$DIR/domain_topic.md"
#     echo "Taxonomy generation completed!"
# else
#     echo "Warning: No survey markdown files found in $DIR/md_surveys, skipping taxonomy generation."
# fi

# Classify citation intents
mkdir -p "$DIR/auto_intent_class_raw"

if [ -d "$DIR/md_papers" ] && [ -d "$DIR/yaml_refs" ]; then
    echo "Classifying citation intents..."

    paper_count=0
    for md_file in "$DIR/md_papers"/*.md; do
        [ -f "$md_file" ] || continue

        filename=$(basename "$md_file" .md)
        yaml_ref="$DIR/yaml_refs/${filename}.yaml"

        # Check if corresponding yaml exists
        if [ ! -f "$yaml_ref" ]; then
            echo "  Warning: No yaml_ref found for $filename, skipping..."
            continue
        fi

        output_yaml="$DIR/auto_intent_class_raw/${filename}.yaml"

        echo "  Processing: $filename"
        python -m data.raw.util.classify_intent \
            --md_paper_path "$md_file" \
            --yaml_ref_path "$yaml_ref" \
            --intent_taxo_path "data/raw/general_intent.md" \
            --output_yaml_path "$output_yaml"

        ((paper_count++))
    done

    if [ $paper_count -gt 0 ]; then
        echo "Intent classification completed for $paper_count papers!"
    else
        echo "No papers found to classify."
    fi
else
    echo "Warning: md_papers or yaml_refs directory not found, skipping intent classification."
fi

# Classify citation topics
mkdir -p "$DIR/auto_topic_class_raw"

if [ -d "$DIR/md_papers" ] && [ -d "$DIR/yaml_refs" ] && [ -f "$DIR/domain_topic.md" ]; then
    echo "Classifying citation topics..."

    paper_count=0
    for md_file in "$DIR/md_papers"/*.md; do
        [ -f "$md_file" ] || continue

        filename=$(basename "$md_file" .md)
        yaml_ref="$DIR/yaml_refs/${filename}.yaml"

        # Check if corresponding yaml exists
        if [ ! -f "$yaml_ref" ]; then
            echo "  Warning: No yaml_ref found for $filename, skipping..."
            continue
        fi

        output_yaml="$DIR/auto_topic_class_raw/${filename}.yaml"

        echo "  Processing: $filename"
        python -m data.raw.util.classify_topic \
            --md_paper_path "$md_file" \
            --yaml_ref_path "$yaml_ref" \
            --topic_taxo_path "$DIR/domain_topic.md" \
            --output_yaml_path "$output_yaml"

        ((paper_count++))
    done

    if [ $paper_count -gt 0 ]; then
        echo "Topic classification completed for $paper_count papers!"
    else
        echo "No papers found to classify."
    fi
else
    echo "Warning: md_papers, yaml_refs, or domain_topic.md not found, skipping topic classification."
fi
