#!/bin/bash

# Generate topic taxonomy from survey papers
# Usage: generate_taxonomy.sh <directory>

DIR="$1"

echo "=== Step 4: Taxonomy Generation ==="

# Generate topic name from directory name
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
