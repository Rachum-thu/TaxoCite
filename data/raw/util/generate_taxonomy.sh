#!/bin/bash

# Generate topic taxonomy from survey papers
# Usage: generate_taxonomy.sh <directory> [--topic_name <topic_name>]

DIR="$1"
TOPIC_NAME=""

echo "=== Step 4: Taxonomy Generation ==="

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --topic_name)
            TOPIC_NAME="$2"
            shift 2
            ;;
        *)
            DIR="$1"
            shift
            ;;
    esac
done

# Check if topic name is provided, else use directory name
if [ -z "$TOPIC_NAME" ]; then
    TOPIC_NAME=$(basename "$DIR" | tr '_' ' ')
fi

# # Generate topic name from directory name
# TOPIC_NAME=$(basename "$DIR" | tr '_' ' ')

if [ -d "$DIR/md_surveys" ] && [ "$(ls -A "$DIR/md_surveys"/*.md 2>/dev/null)" ]; then
    echo "Generating taxonomy for topic: $TOPIC_NAME"
    python -m data.raw.util.surveys2taxo \
        --survey_dir "$DIR/md_surveys" \
        --paper_dir "$DIR/md_papers" \
        --topic_name "$TOPIC_NAME" \
        --output "$DIR/domain_topic.md"
    echo "Taxonomy generation completed!"
else
    echo "Warning: No survey or paper markdown files found in $DIR/md_surveys or $DIR/md_papers, skipping taxonomy generation."
fi
