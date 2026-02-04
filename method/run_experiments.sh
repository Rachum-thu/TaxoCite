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
PARALLEL_JOBS=16  # Number of parallel jobs

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
echo "Parallel jobs: $PARALLEL_JOBS"
echo "Methods: zeroshot, cot, reflexion, taxocite"
echo "=========================================="

# Create output directories for all methods
mkdir -p "$DIR/result/zeroshot_$MODEL"
mkdir -p "$DIR/result/cot_$MODEL"
mkdir -p "$DIR/result/reflexion_$MODEL"
mkdir -p "$DIR/result/taxo_reverse_$MODEL"
mkdir -p "$DIR/result/taxo_cot_k_$MODEL"
mkdir -p "$DIR/result/taxo_final_$MODEL"

# Export variables for parallel
export DIR MODEL

# ==========================================
# Method 1: Zero-shot Classification
# ==========================================
echo ""
echo "=== Method 1/4: Zero-shot Classification (parallel: $PARALLEL_JOBS) ==="

run_zeroshot() {
    seg_file="$1"
    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    output_file="$DIR/result/zeroshot_$MODEL/${filename}.yaml"

    if [ ! -f "$ref_file" ]; then
        echo "Warning: No ref file for $filename, skipping..."
        return 1
    fi

    echo "  Processing $filename..."
    python -m method.zeroshot \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL"
}
export -f run_zeroshot

find "$DIR/seg_papers" -name "*.yaml" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_zeroshot "$@"' _ {}

echo "Zero-shot completed"

# ==========================================
# Method 2: Chain-of-Thought Classification
# ==========================================
echo ""
echo "=== Method 2/4: Chain-of-Thought (CoT) Classification (parallel: $PARALLEL_JOBS) ==="

run_cot() {
    seg_file="$1"
    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    output_file="$DIR/result/cot_$MODEL/${filename}.yaml"

    if [ ! -f "$ref_file" ]; then
        echo "Warning: No ref file for $filename, skipping..."
        return 1
    fi

    echo "  Processing $filename..."
    python -m method.cot \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL"
}
export -f run_cot

find "$DIR/seg_papers" -name "*.yaml" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_cot "$@"' _ {}

echo "CoT completed"

# ==========================================
# Method 3: Reflexion Classification (reuses zeroshot results)
# ==========================================
echo ""
echo "=== Method 3/4: Reflexion Classification (parallel: $PARALLEL_JOBS) ==="

run_reflexion() {
    seg_file="$1"
    filename=$(basename "$seg_file" .yaml)
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    zeroshot_file="$DIR/result/zeroshot_$MODEL/${filename}.yaml"
    output_file="$DIR/result/reflexion_$MODEL/${filename}.yaml"

    if [ ! -f "$ref_file" ]; then
        echo "Warning: No ref file for $filename, skipping..."
        return 1
    fi

    if [ ! -f "$zeroshot_file" ]; then
        echo "Warning: No zeroshot result for $filename, skipping..."
        return 1
    fi

    echo "  Processing $filename..."
    python -m method.reflexion \
        --general_intent_taxonomy data/raw/general_intent.md \
        --domain_topic_taxonomy "$DIR/domain_topic.md" \
        --seg_paper "$seg_file" \
        --ref_list "$ref_file" \
        --output "$output_file" \
        --model "$MODEL" \
        --zeroshot_result "$zeroshot_file"
}
export -f run_reflexion

find "$DIR/seg_papers" -name "*.yaml" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_reflexion "$@"' _ {}

echo "Reflexion completed"

# ==========================================
# Method 4: TaxoCite (depends on CoT results)
# ==========================================
echo ""
echo "=== Method 4/4: TaxoCite Classification (parallel: $PARALLEL_JOBS) ==="

run_taxocite() {
    cot_file="$1"
    filename=$(basename "$cot_file" .yaml)
    seg_file="$DIR/seg_papers/${filename}.yaml"
    ref_file="$DIR/yaml_refs/${filename}.yaml"
    reverse_output="$DIR/result/taxo_reverse_$MODEL/${filename}.yaml"
    cot_k_output="$DIR/result/taxo_cot_k_$MODEL/${filename}.yaml"
    final_output="$DIR/result/taxo_final_$MODEL/${filename}.yaml"

    if [ ! -f "$seg_file" ] || [ ! -f "$ref_file" ]; then
        echo "Warning: Missing seg_paper or ref_list for $filename, skipping..."
        return 1
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
}
export -f run_taxocite

find "$DIR/result/cot_$MODEL" -name "*.yaml" | xargs -P $PARALLEL_JOBS -I {} bash -c 'run_taxocite "$@"' _ {}

echo "TaxoCite completed"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Experimental Comparison Completed!"
echo "=========================================="
echo "Results saved to:"
echo "  1. Zero-shot:   $DIR/result/zeroshot_$MODEL/"
echo "  2. CoT:         $DIR/result/cot_$MODEL/"
echo "  3. Reflexion:   $DIR/result/reflexion_$MODEL/"
echo "  4. TaxoCite:    $DIR/result/taxo_final_$MODEL/"
echo "=========================================="
