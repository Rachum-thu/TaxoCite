"""
Check label distribution across annotated citations.

This script reads all human_todo YAML files and computes statistics
for intent and topic labels, displaying them within the taxonomy structure.
"""

import argparse
import yaml
from pathlib import Path
from collections import Counter
from typing import Tuple
from method.utils import extract_taxonomy_nodes, extract_node_definition


def collect_labels_from_human_todo(human_todo_dir: Path) -> Tuple[Counter, Counter]:
    """
    Collect all intent and topic labels from human_todo directory.

    Args:
        human_todo_dir: Path to human_todo directory

    Returns:
        Tuple of (intent_counter, topic_counter)
    """
    intent_counter = Counter()
    topic_counter = Counter()

    # Read all yaml files
    for yaml_file in human_todo_dir.glob("*.yaml"):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Extract citations from all blocks
        for block in data.get("blocks", []):
            for citation in block.get("citations", []):
                intent_label = citation.get("intent_label")
                topic_label = citation.get("topic_label")

                if intent_label:
                    intent_counter[intent_label] += 1
                if topic_label:
                    topic_counter[topic_label] += 1

    return intent_counter, topic_counter




def format_taxonomy_with_stats(
    taxonomy_text: str,
    label_counter: Counter,
    label_type: str
) -> str:
    """
    Format taxonomy with statistics, showing only non-zero nodes sorted by count.

    Args:
        taxonomy_text: Original taxonomy markdown
        label_counter: Counter of labels
        label_type: "Intent" or "Topic"

    Returns:
        Formatted string
    """
    # Extract all nodes from taxonomy using existing utility
    nodes = extract_taxonomy_nodes(taxonomy_text)
    total_count = sum(label_counter.values())

    output_lines = []

    # Add header
    output_lines.append(f"{'=' * 80}")
    output_lines.append(f"{label_type} Taxonomy Distribution")
    output_lines.append(f"Total {label_type} Citations: {total_count}")
    output_lines.append(f"{'=' * 80}\n")

    # Track which labels are in taxonomy
    labels_in_taxonomy = set(nodes)

    # Collect non-zero nodes with their counts
    node_stats = []
    for node_name in nodes:
        count = label_counter.get(node_name, 0)
        if count > 0:
            percentage = (count / total_count * 100) if total_count > 0 else 0
            description = extract_node_definition(taxonomy_text, node_name)
            node_stats.append((node_name, count, percentage, description))

    # Sort by count (descending)
    node_stats.sort(key=lambda x: x[1], reverse=True)

    # Output sorted nodes
    for node_name, count, percentage, description in node_stats:
        output_lines.append(f"{node_name}: {count} citations ({percentage:.2f}%)")
        if description:
            # Show first line of description
            first_line = description.split('\n')[0].strip()
            if first_line.startswith('**Description:**'):
                first_line = first_line.replace('**Description:**', '').strip()
            output_lines.append(f"  → {first_line}")
        output_lines.append("")

    # Add "Other" labels that are not in taxonomy
    other_labels = {label: count for label, count in label_counter.items()
                    if label not in labels_in_taxonomy}

    if other_labels:
        output_lines.append(f"{'-' * 80}")
        output_lines.append("Other Labels (Not in Taxonomy)")
        output_lines.append(f"{'-' * 80}\n")

        for label, count in sorted(other_labels.items(), key=lambda x: -x[1]):
            percentage = (count / total_count * 100) if total_count > 0 else 0
            output_lines.append(f"{label}: {count} citations ({percentage:.2f}%)")
        output_lines.append("")

    # Summary
    non_zero_in_taxonomy = len(node_stats)
    non_zero_other = len(other_labels)
    total_nodes = len(nodes)

    output_lines.append(f"{'-' * 80}")
    output_lines.append("Summary:")
    output_lines.append(f"  Total citations: {total_count}")
    output_lines.append(f"  Total nodes in taxonomy: {total_nodes}")
    output_lines.append(f"  Non-zero nodes in taxonomy: {non_zero_in_taxonomy}")
    output_lines.append(f"  Other labels (not in taxonomy): {non_zero_other}")
    output_lines.append(f"{'=' * 80}\n")

    return '\n'.join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check label distribution in human_todo annotations"
    )
    parser.add_argument(
        "--intent_taxonomy",
        required=True,
        help="Path to intent taxonomy markdown file"
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset directory (e.g., data/raw/gis/climate)"
    )

    args = parser.parse_args()

    # Construct paths
    dataset_dir = Path(args.dataset_dir)
    human_todo_dir = dataset_dir / "human_todo"
    topic_taxonomy_path = dataset_dir / "domain_topic.md"

    if not human_todo_dir.exists():
        print(f"Error: {human_todo_dir} does not exist")
        return

    if not topic_taxonomy_path.exists():
        print(f"Error: {topic_taxonomy_path} does not exist")
        return

    # Collect labels
    print(f"Collecting labels from {human_todo_dir}...")
    intent_counter, topic_counter = collect_labels_from_human_todo(human_todo_dir)

    print(f"Found {sum(intent_counter.values())} intent labels, {sum(topic_counter.values())} topic labels")
    print()

    # Read taxonomies
    with open(args.intent_taxonomy, 'r', encoding='utf-8') as f:
        intent_taxonomy = f.read()

    with open(topic_taxonomy_path, 'r', encoding='utf-8') as f:
        topic_taxonomy = f.read()

    # Format and print intent distribution
    print("=" * 80)
    print(format_taxonomy_with_stats(intent_taxonomy, intent_counter, "Intent"))
    print()

    # Format and print topic distribution
    print("=" * 80)
    print(format_taxonomy_with_stats(topic_taxonomy, topic_counter, "Topic"))


if __name__ == "__main__":
    main()
