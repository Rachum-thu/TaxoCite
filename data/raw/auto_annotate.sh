#!/bin/bash

# Automatic annotation pipeline using CoT and TaxoCite
# Usage: auto_annotate.sh <directory_path> [--model <model_name>]
# Default model: gpt-5-2025-08-07

# Parse arguments
DIR=""
MODEL="gpt-5-2025-08-07"  # Default model

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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

# Check if required arguments are provided
if [ -z "$DIR" ]; then
    echo "Usage: $0 <directory_path> [--model <model_name>]"
    echo "Example: $0 data/raw/cs/retrieval_augmented_generation"
    echo "Example: $0 data/raw/cs/retrieval_augmented_generation --model gpt-4o-mini-2025-07-18"
    echo "Default model: gpt-5-2025-08-07"
    exit 1
fi

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Check if required files/directories exist
if [ ! -f "data/raw/general_intent.md" ]; then
    echo "Error: data/raw/general_intent.md not found"
    exit 1
fi

if [ ! -f "$DIR/domain_topic.md" ]; then
    echo "Error: $DIR/domain_topic.md not found"
    exit 1
fi

if [ ! -d "$DIR/seg_papers" ]; then
    echo "Error: $DIR/seg_papers directory not found"
    exit 1
fi

if [ ! -d "$DIR/yaml_refs" ]; then
    echo "Error: $DIR/yaml_refs directory not found"
    exit 1
fi

echo "=========================================="
echo "Processing directory: $DIR"
echo "Model: $MODEL"
echo "=========================================="

# Create output directory with model suffix under result/
mkdir -p "$DIR/result/cot_output_$MODEL"

# Step 1: Run CoT on all papers
echo "=== Step 1: Chain-of-Thought Classification ==="

cot_count=0
for seg_file in "$DIR/seg_papers"/*.yaml; do
    [ -f "$seg_file" ] || continue

    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    output_file="$DIR/result/cot_output_$MODEL/${filename}.yaml"

    # Check if corresponding ref file exists
    if [ ! -f "$ref_file" ]; then
        echo "Warning: No corresponding ref file for $filename, skipping..."
        continue
    fi

    echo "Processing $filename..."
    python -m method.cot \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL"

    if [ $? -eq 0 ]; then
        ((cot_count++))
    else
        echo "Error: Failed to process $filename with CoT"
    fi
done

echo "CoT classification completed for $cot_count papers!"

# Step 2: Run TaxoCite on all CoT results
echo ""
echo "=== Step 2: TaxoCite Classification ==="

# Create output directories with model suffix under result/
mkdir -p "$DIR/result/taxo_reverse_$MODEL"
mkdir -p "$DIR/result/taxo_cot_k_$MODEL"
mkdir -p "$DIR/result/taxo_final_$MODEL"

taxocite_count=0
for cot_file in "$DIR/result/cot_output_$MODEL"/*.yaml; do
    [ -f "$cot_file" ] || continue

    filename=$(basename "$cot_file" .yaml)
    seg_file="$DIR/seg_papers/${filename}.yaml"
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    reverse_output="$DIR/result/taxo_reverse_$MODEL/${filename}.yaml"
    cot_k_output="$DIR/result/taxo_cot_k_$MODEL/${filename}.yaml"
    final_output="$DIR/result/taxo_final_$MODEL/${filename}.yaml"

    # Check if corresponding files exist
    if [ ! -f "$seg_file" ] || [ ! -f "$ref_file" ]; then
        echo "Warning: Missing seg_paper or ref_list for $filename, skipping..."
        continue
    fi

    echo "Processing $filename with TaxoCite..."
    python -m method.taxocite \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --cot_result "$cot_file" \
        --reverse_output "$reverse_output" \
        --cot_k_output "$cot_k_output" \
        --final_output "$final_output" \
        --model "$MODEL"

    if [ $? -eq 0 ]; then
        ((taxocite_count++))
    else
        echo "Error: Failed to process $filename with TaxoCite"
    fi
done

echo "TaxoCite classification completed for $taxocite_count papers!"

# Step 3: Convert to human-readable format
echo ""
echo "=== Step 3: Convert to Human-Readable Format ==="

# Create human_todo directory
mkdir -p "$DIR/human_todo"

convert_count=0
for annotated_file in "$DIR/result/taxo_final_$MODEL"/*.yaml; do
    [ -f "$annotated_file" ] || continue

    filename=$(basename "$annotated_file" .yaml)
    seg_file="$DIR/seg_papers/${filename}.yaml"
    output_file="$DIR/human_todo/${filename}.yaml"

    # Check if corresponding seg_paper exists
    if [ ! -f "$seg_file" ]; then
        echo "Warning: Missing seg_paper for $filename, skipping..."
        continue
    fi

    echo "Converting $filename to human-readable format..."
    python -m data.raw.util.convert4human \
        --annotated_result "$annotated_file" \
        --seg_paper "$seg_file" \
        --output "$output_file"

    if [ $? -eq 0 ]; then
        ((convert_count++))
    else
        echo "Error: Failed to convert $filename"
    fi
done

echo "Converted $convert_count papers to human-readable format!"

echo "=========================================="
echo "Annotation pipeline completed!"
echo "Summary:"
echo "  - CoT processed: $cot_count papers"
echo "  - TaxoCite processed: $taxocite_count papers"
echo "  - Converted for human review: $convert_count papers"
echo "=========================================="
