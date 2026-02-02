"""
Convert annotated results to human-readable format for manual review.

This script combines segmented paper blocks with their annotations,
creating a structured YAML output that shows each block's content
along with its citations and labels.
"""

import argparse
import re
import yaml
from typing import List, Dict, Any


def extract_blocks_from_segmented_markdown(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract all blocks from segmented markdown.

    Args:
        markdown: The segmented markdown text containing <block id="X">...</block> tags

    Returns:
        List of dictionaries with block_id and content
    """
    blocks = []
    # Pattern to match <block id="X">content</block>
    pattern = r'<block id="(\d+)">(.*?)</block>'

    for match in re.finditer(pattern, markdown, re.DOTALL):
        block_id = int(match.group(1))
        content = match.group(2).strip()
        blocks.append({
            "block_id": block_id,
            "content": content
        })

    # Sort by block_id to ensure correct order
    blocks.sort(key=lambda x: x["block_id"])
    return blocks


def build_block_citations_map(citations: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, str]]]:
    """
    Build a mapping from block_id to citations with labels.

    Args:
        citations: List of citation dictionaries from annotated result

    Returns:
        Dictionary mapping block_id to list of citations with their labels
    """
    block_citations_map = {}

    for citation in citations:
        marker = citation["unique_context_marker"]
        block_ids = citation["block_ids"]
        intent_labels = citation["intent_labels"]
        topic_labels = citation["topic_labels"]

        # Each citation can appear in multiple blocks
        for idx, block_id in enumerate(block_ids):
            if block_id not in block_citations_map:
                block_citations_map[block_id] = []

            block_citations_map[block_id].append({
                "marker": marker,
                "intent_label": intent_labels[idx],
                "topic_label": topic_labels[idx]
            })

    return block_citations_map


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated results to human-readable format"
    )
    parser.add_argument(
        "--annotated_result",
        required=True,
        help="Path to annotated result YAML file"
    )
    parser.add_argument(
        "--seg_paper",
        required=True,
        help="Path to segmented paper YAML file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output YAML file"
    )

    args = parser.parse_args()

    # Load annotated result
    with open(args.annotated_result, 'r', encoding='utf-8') as f:
        annotated_data = yaml.safe_load(f)

    # Load segmented paper
    with open(args.seg_paper, 'r', encoding='utf-8') as f:
        seg_paper_data = yaml.safe_load(f)

    # Extract blocks from segmented markdown
    blocks = extract_blocks_from_segmented_markdown(
        seg_paper_data["segmented_markdown"]
    )

    # Build block to citations mapping
    block_citations_map = build_block_citations_map(
        annotated_data["citations"]
    )

    # Combine blocks with their citations
    output_blocks = []
    for block in blocks:
        block_id = block["block_id"]
        output_block = {
            "block_id": block_id,
            "content": block["content"],
            "citations": block_citations_map.get(block_id, [])
        }
        output_blocks.append(output_block)

    # Create output structure
    output_data = {
        "title": seg_paper_data.get("title", ""),
        "blocks": output_blocks
    }

    # Write output YAML
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, allow_unicode=True, sort_keys=False, width=120)

    print(f"Successfully converted to human-readable format: {args.output}")
    print(f"Total blocks: {len(output_blocks)}")
    print(f"Blocks with citations: {sum(1 for b in output_blocks if b['citations'])}")


if __name__ == "__main__":
    main()
