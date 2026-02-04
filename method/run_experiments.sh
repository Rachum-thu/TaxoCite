#!/bin/bash

# Experimental comparison of classification methods
# Usage: run_experiments.sh <directory_path> [--model <model_name>]
# Default model: gpt-5-2025-08-07
#
# This script runs all methods (zeroshot, cot, reflexion, taxocite) on the same dataset
# to enable performance comparison

# Parse arguments
DIR=""
MODEL="gpt-5-nano-2025-08-07"  # Default model

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
    echo "Default model: gpt-5-nano-2025-08-07"
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
echo "Running Experimental Comparison"
echo "Directory: $DIR"
echo "Model: $MODEL"
echo "Methods: zeroshot, cot, reflexion, taxocite"
echo "=========================================="

# Create output directories for all methods
mkdir -p "$DIR/result/zeroshot_$MODEL"
mkdir -p "$DIR/result/cot_$MODEL"
mkdir -p "$DIR/result/reflexion_$MODEL"
mkdir -p "$DIR/result/taxo_reverse_$MODEL"
mkdir -p "$DIR/result/taxo_cot_k_$MODEL"
mkdir -p "$DIR/result/taxo_final_$MODEL"

# Initialize counters
zeroshot_count=0
cot_count=0
reflexion_count=0
taxocite_count=0

# ==========================================
# Method 1: Zero-shot Classification
# ==========================================
echo ""
echo "=== Method 1/4: Zero-shot Classification ==="

for seg_file in "$DIR/seg_papers"/*.yaml; do
    [ -f "$seg_file" ] || continue

    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    output_file="$DIR/result/zeroshot_$MODEL/${filename}.yaml"

    # Check if corresponding ref file exists
    if [ ! -f "$ref_file" ]; then
        echo "Warning: No corresponding ref file for $filename, skipping..."
        continue
    fi

    echo "  Processing $filename..."
    python -m method.zeroshot \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL"

    if [ $? -eq 0 ]; then
        ((zeroshot_count++))
    else
        echo "  Error: Failed to process $filename with zeroshot"
    fi
done

echo "Zero-shot completed: $zeroshot_count papers"

# ==========================================
# Method 2: Chain-of-Thought Classification
# ==========================================
echo ""
echo "=== Method 2/4: Chain-of-Thought (CoT) Classification ==="

for seg_file in "$DIR/seg_papers"/*.yaml; do
    [ -f "$seg_file" ] || continue

    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    output_file="$DIR/result/cot_$MODEL/${filename}.yaml"

    # Check if corresponding ref file exists
    if [ ! -f "$ref_file" ]; then
        echo "Warning: No corresponding ref file for $filename, skipping..."
        continue
    fi

    echo "  Processing $filename..."
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
        echo "  Error: Failed to process $filename with CoT"
    fi
done

echo "CoT completed: $cot_count papers"

# ==========================================
# Method 3: Reflexion Classification (reuses zeroshot results)
# ==========================================
echo ""
echo "=== Method 3/4: Reflexion Classification (reusing zeroshot) ==="

for seg_file in "$DIR/seg_papers"/*.yaml; do
    [ -f "$seg_file" ] || continue

    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    zeroshot_file="$DIR/result/zeroshot_$MODEL/${filename}.yaml"
    output_file="$DIR/result/reflexion_$MODEL/${filename}.yaml"

    # Check if corresponding files exist
    if [ ! -f "$ref_file" ]; then
        echo "Warning: No corresponding ref file for $filename, skipping..."
        continue
    fi

    if [ ! -f "$zeroshot_file" ]; then
        echo "Warning: No zeroshot result for $filename, skipping..."
        continue
    fi

    echo "  Processing $filename (reusing zeroshot result)..."
    python -m method.reflexion \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL" \
        --zeroshot_result "$zeroshot_file"

    if [ $? -eq 0 ]; then
        ((reflexion_count++))
    else
        echo "  Error: Failed to process $filename with reflexion"
    fi
done

echo "Reflexion completed: $reflexion_count papers"

# ==========================================
# Method 4: TaxoCite (depends on CoT results)
# ==========================================
echo ""
echo "=== Method 4/4: TaxoCite Classification ==="

for cot_file in "$DIR/result/cot_$MODEL"/*.yaml; do
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

    echo "  Processing $filename..."
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
        echo "  Error: Failed to process $filename with TaxoCite"
    fi
done

echo "TaxoCite completed: $taxocite_count papers"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Experimental Comparison Completed!"
echo "=========================================="
echo "Results Summary:"
echo "  1. Zero-shot:   $zeroshot_count papers   -> $DIR/result/zeroshot_$MODEL/"
echo "  2. CoT:         $cot_count papers   -> $DIR/result/cot_$MODEL/"
echo "  3. Reflexion:   $reflexion_count papers   -> $DIR/result/reflexion_$MODEL/"
echo "  4. TaxoCite:    $taxocite_count papers   -> $DIR/result/taxo_final_$MODEL/"
echo "=========================================="
echo ""
echo "Next steps for comparison:"
echo "  - Compare outputs across result/<method>_$MODEL/ directories"
echo "  - Run evaluation metrics on each method's results"
echo "  - Analyze method-specific strengths and weaknesses"
echo "=========================================="
