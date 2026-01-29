import argparse
import yaml
import random
import copy
from collections import Counter
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from method.prompts import cot_intent_prompt, cot_topic_prompt, taxocite_reverse_intent_prompt, taxocite_reverse_topic_prompt
from method.utils import (QuotedDumper, extract_blocks, extract_taxonomy_nodes, validate_label,
                          mask_segmented_markdown, extract_node_definition, fill_labels_by_marker,
                          build_block_to_citations)


class Citation(BaseModel):
    marker: str = Field(description="The citation marker, e.g., '[13]'.")
    analysis: str = Field(
        description=(
            "one-sentence reasoning about the marker's role/topic in the current block. "
        )
    )
    label: str = Field(
        description=(
            "The EXACT NAME of ONE taxonomy node (copy the node title verbatim). "
        )
    )


class CitationClassification(BaseModel):
    classifications: list[Citation]


class NodeAssignment(BaseModel):
    node: str = Field(description="The node name.")
    reasoning: str = Field(description="One sentence explaining why these markers belong to this node.")
    markers: list[str] = Field(description="List of markers that belong to this node. Can be empty if no markers match.")


class ReverseRetrievalResult(BaseModel):
    assignments: list[NodeAssignment]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_intent_taxonomy", required=True)
    parser.add_argument("--domain_topic_taxonomy", required=True)
    parser.add_argument("--seg_paper", required=True)
    parser.add_argument("--ref_list", required=True)
    parser.add_argument("--cot_result", required=True)
    parser.add_argument("--reverse_output", required=True)
    parser.add_argument("--cot_k_output", required=True)
    parser.add_argument("--final_output", required=True)
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    # Read inputs
    with open(args.general_intent_taxonomy) as f:
        intent_taxonomy = f.read()
    with open(args.domain_topic_taxonomy) as f:
        topic_taxonomy = f.read()

    seg_data = yaml.safe_load(open(args.seg_paper))
    ref_data = yaml.safe_load(open(args.ref_list))
    cot_data = yaml.safe_load(open(args.cot_result))

    abstract = seg_data.get("abstract", "")
    segmented_markdown = seg_data.get("segmented_markdown", "")
    blocks = extract_blocks(segmented_markdown)
    citations = ref_data.get("citations", [])

    # Extract valid nodes from taxonomies
    valid_intent_nodes = extract_taxonomy_nodes(intent_taxonomy)
    valid_topic_nodes = extract_taxonomy_nodes(topic_taxonomy)

    print(f"Found {len(valid_intent_nodes)} intent nodes, {len(valid_topic_nodes)} topic nodes")

    # Initialize labels
    for citation in citations:
        citation["intent_labels"] = [None] * len(citation.get("block_ids", []))
        citation["topic_labels"] = [None] * len(citation.get("block_ids", []))

    # Build block_id to citations mapping
    block_to_citations = build_block_to_citations(citations)

    # Build CoT predictions mapping
    cot_citations = cot_data.get("citations", [])
    cot_marker_to_citation = {c.get("unique_context_marker", ""): c for c in cot_citations}

    llm = ChatOpenAI(model=args.model, reasoning_effort="medium")
    reverse_llm = llm.with_structured_output(ReverseRetrievalResult)
    cot_llm = llm.with_structured_output(CitationClassification)

    print("Phase 1: Reverse retrieval...")

    # Collect all reverse retrieval prompts
    intent_prompts, intent_metadata = [], []
    topic_prompts, topic_metadata = [], []

    for block_id in range(len(blocks)):
        if block_id not in block_to_citations:
            continue

        block_citations = block_to_citations[block_id]
        markers = [bc["marker"] for bc in block_citations]
        masked_md = mask_segmented_markdown(segmented_markdown, block_id)

        # Collect CoT predictions for this block
        intent_nodes_in_block, topic_nodes_in_block = set(), set()
        for bc in block_citations:
            cot_citation = cot_marker_to_citation.get(bc["marker"])
            if cot_citation:
                labels = cot_citation.get("intent_labels", [])
                if bc["idx"] < len(labels) and labels[bc["idx"]] and labels[bc["idx"]] != "Other Intent":
                    intent_nodes_in_block.add(labels[bc["idx"]])
                labels = cot_citation.get("topic_labels", [])
                if bc["idx"] < len(labels) and labels[bc["idx"]] and labels[bc["idx"]] != "Other Topic":
                    topic_nodes_in_block.add(labels[bc["idx"]])

        # Prepare intent prompt
        if intent_nodes_in_block:
            nodes_with_defs = [f"- {node}:\n{extract_node_definition(intent_taxonomy, node)}"
                              for node in intent_nodes_in_block]
            intent_prompts.append(taxocite_reverse_intent_prompt.format(
                abstract=abstract, masked_markdown=masked_md, markers=', '.join(markers),
                nodes_with_definitions='\n\n'.join(nodes_with_defs)
            ))
            intent_metadata.append((block_citations, valid_intent_nodes))

        # Prepare topic prompt
        if topic_nodes_in_block:
            nodes_with_defs = [f"- {node}:\n{extract_node_definition(topic_taxonomy, node)}"
                              for node in topic_nodes_in_block]
            topic_prompts.append(taxocite_reverse_topic_prompt.format(
                abstract=abstract, masked_markdown=masked_md, markers=', '.join(markers),
                nodes_with_definitions='\n\n'.join(nodes_with_defs)
            ))
            topic_metadata.append((block_citations, valid_topic_nodes))

    # Batch process reverse retrieval
    if intent_prompts:
        intent_results = reverse_llm.batch(intent_prompts, config={"max_concurrency": 32})
        for result, (block_citations, valid_nodes) in zip(intent_results, intent_metadata):
            for assignment in result.assignments:
                validated_node = validate_label(assignment.node, valid_nodes, "Other Intent")
                for marker in assignment.markers:
                    fill_labels_by_marker(block_citations, marker, validated_node, "intent_labels")

    if topic_prompts:
        topic_results = reverse_llm.batch(topic_prompts, config={"max_concurrency": 32})
        for result, (block_citations, valid_nodes) in zip(topic_results, topic_metadata):
            for assignment in result.assignments:
                validated_node = validate_label(assignment.node, valid_nodes, "Other Topic")
                for marker in assignment.markers:
                    fill_labels_by_marker(block_citations, marker, validated_node, "topic_labels")

    # Save Phase 1 (reverse retrieval) results
    print("Saving reverse retrieval results...")
    reverse_citations = copy.deepcopy(citations)
    with open(args.reverse_output, "w") as f:
        yaml.dump({"citations": reverse_citations}, f, Dumper=QuotedDumper, allow_unicode=True, sort_keys=False, width=1000000)

    print("Phase 2: Self-consistency for unmatched markers...")

    # Initialize cot_k_citations to store k predictions for each marker
    cot_k_citations = copy.deepcopy(citations)
    for citation in cot_k_citations:
        # Change to list[list[str]] - outer list for blocks, inner list for k predictions
        citation["intent_labels"] = [[] for _ in citation.get("block_ids", [])]
        citation["topic_labels"] = [[] for _ in citation.get("block_ids", [])]

    # Build block_to_cot_k mapping
    block_to_cot_k = build_block_to_citations(cot_k_citations)

    # Helper function to get CoT label
    def get_cot_label(marker, idx, label_type):
        cot_citation = cot_marker_to_citation.get(marker)
        if not cot_citation:
            return None
        labels = cot_citation.get(label_type, [])
        return labels[idx] if idx < len(labels) else None

    # Collect all self-consistency prompts
    intent_prompts, intent_metadata = [], []
    topic_prompts, topic_metadata = [], []

    for block_id in range(len(blocks)):
        if block_id not in block_to_citations:
            continue

        block_citations = block_to_citations[block_id]
        masked_md = mask_segmented_markdown(segmented_markdown, block_id)

        # Find mismatched markers
        unmatched_intent_markers = [
            bc["marker"] for bc in block_citations
            if bc["citation"]["intent_labels"][bc["idx"]] != get_cot_label(bc["marker"], bc["idx"], "intent_labels")
        ]
        unmatched_topic_markers = [
            bc["marker"] for bc in block_citations
            if bc["citation"]["topic_labels"][bc["idx"]] != get_cot_label(bc["marker"], bc["idx"], "topic_labels")
        ]

        # Prepare intent prompts (k times for each block)
        if unmatched_intent_markers:
            prompt = cot_intent_prompt.format(
                markers=', '.join(unmatched_intent_markers), taxonomy=intent_taxonomy,
                abstract=abstract, masked_markdown=masked_md
            )
            for _ in range(args.k):
                intent_prompts.append(prompt)
                intent_metadata.append((block_id, unmatched_intent_markers, block_citations))

        # Prepare topic prompts (k times for each block)
        if unmatched_topic_markers:
            prompt = cot_topic_prompt.format(
                markers=', '.join(unmatched_topic_markers), taxonomy=topic_taxonomy,
                abstract=abstract, masked_markdown=masked_md
            )
            for _ in range(args.k):
                topic_prompts.append(prompt)
                topic_metadata.append((block_id, unmatched_topic_markers, block_citations))

    # Batch process self-consistency
    if intent_prompts:
        intent_results = cot_llm.batch(intent_prompts, config={"max_concurrency": 32})
        # Group results by (block_id, marker)
        votes_by_marker = {}
        for result, (block_id, markers, block_citations) in zip(intent_results, intent_metadata):
            for c in result.classifications:
                key = (block_id, c.marker)
                if key not in votes_by_marker:
                    votes_by_marker[key] = []
                votes_by_marker[key].append(validate_label(c.label, valid_intent_nodes, "Other Intent"))

        # Majority vote and fill results
        for (block_id, marker), votes in votes_by_marker.items():
            fill_labels_by_marker(block_to_cot_k[block_id], marker, votes, "intent_labels")
            counter = Counter(votes)
            max_count = counter.most_common(1)[0][1]
            top_choices = [label for label, count in counter.items() if count == max_count]
            final_label = random.choice(top_choices)
            fill_labels_by_marker(block_to_citations[block_id], marker, final_label, "intent_labels")

    if topic_prompts:
        topic_results = cot_llm.batch(topic_prompts, config={"max_concurrency": 32})
        # Group results by (block_id, marker)
        votes_by_marker = {}
        for result, (block_id, markers, block_citations) in zip(topic_results, topic_metadata):
            for c in result.classifications:
                key = (block_id, c.marker)
                if key not in votes_by_marker:
                    votes_by_marker[key] = []
                votes_by_marker[key].append(validate_label(c.label, valid_topic_nodes, "Other Topic"))

        # Majority vote and fill results
        for (block_id, marker), votes in votes_by_marker.items():
            fill_labels_by_marker(block_to_cot_k[block_id], marker, votes, "topic_labels")
            counter = Counter(votes)
            max_count = counter.most_common(1)[0][1]
            top_choices = [label for label, count in counter.items() if count == max_count]
            final_label = random.choice(top_choices)
            fill_labels_by_marker(block_to_citations[block_id], marker, final_label, "topic_labels")

    # Write outputs
    print("Saving CoT k-times predictions...")
    with open(args.cot_k_output, "w") as f:
        yaml.dump({"citations": cot_k_citations}, f, Dumper=QuotedDumper, allow_unicode=True, sort_keys=False, width=1000000)

    print("Saving final output...")
    with open(args.final_output, "w") as f:
        yaml.dump({"citations": citations}, f, Dumper=QuotedDumper, allow_unicode=True, sort_keys=False, width=1000000)

    print("Done!")


if __name__ == "__main__":
    main()
