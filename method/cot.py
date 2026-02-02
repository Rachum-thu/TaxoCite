import argparse
import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from method.prompts import cot_intent_prompt, cot_topic_prompt
from method.utils import QuotedDumper, extract_blocks, extract_taxonomy_nodes, validate_label, mask_segmented_markdown


from pydantic import BaseModel, Field


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_intent_taxonomy", required=True)
    parser.add_argument("--domain_topic_taxonomy", required=True)
    parser.add_argument("--seg_paper", required=True)
    parser.add_argument("--ref_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    args = parser.parse_args()

    # Read inputs
    with open(args.general_intent_taxonomy) as f:
        intent_taxonomy = f.read()
    with open(args.domain_topic_taxonomy) as f:
        topic_taxonomy = f.read()

    seg_data = yaml.safe_load(open(args.seg_paper))
    ref_data = yaml.safe_load(open(args.ref_list))

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
    block_to_citations = {}
    for citation in citations:
        marker = citation.get("unique_context_marker", "")
        for idx, block_id in enumerate(citation.get("block_ids", [])):
            if block_id not in block_to_citations:
                block_to_citations[block_id] = []
            block_to_citations[block_id].append({"marker": marker, "citation": citation, "idx": idx})

    # Prepare all prompts
    intent_prompts = []
    topic_prompts = []
    block_ids_to_process = []

    for block_id in range(len(blocks)):
        if block_id not in block_to_citations:
            continue

        block_citations = block_to_citations[block_id]
        markers = [bc["marker"] for bc in block_citations]

        # Generate masked markdown for this block
        masked_md = mask_segmented_markdown(segmented_markdown, block_id)

        intent_prompts.append(cot_intent_prompt.format(
            markers=', '.join(markers),
            taxonomy=intent_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md
        ))

        topic_prompts.append(cot_topic_prompt.format(
            markers=', '.join(markers),
            taxonomy=topic_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md
        ))

        block_ids_to_process.append(block_id)

    # Batch process with LLM
    llm = ChatOpenAI(model=args.model, reasoning_effort="medium")
    structured_llm = llm.with_structured_output(CitationClassification)

    intent_outputs = structured_llm.batch(intent_prompts, config={"max_concurrency": 32})
    topic_outputs = structured_llm.batch(topic_prompts, config={"max_concurrency": 32})

    # Fill results with validation
    for block_id, intent_output, topic_output in zip(block_ids_to_process, intent_outputs, topic_outputs):
        block_citations = block_to_citations[block_id]
        intent_map = {c.marker: validate_label(c.label, valid_intent_nodes, "Other Intent") for c in intent_output.classifications}
        topic_map = {c.marker: validate_label(c.label, valid_topic_nodes, "Other Topic") for c in topic_output.classifications}

        for bc in block_citations:
            marker = bc["marker"]
            citation = bc["citation"]
            idx = bc["idx"]
            citation["intent_labels"][idx] = intent_map.get(marker, "Other Intent")
            citation["topic_labels"][idx] = topic_map.get(marker, "Other Topic")

    # Write output
    with open(args.output, "w") as f:
        yaml.dump({"citations": citations}, f, Dumper=QuotedDumper, allow_unicode=True, sort_keys=False, width=1000000)


if __name__ == "__main__":
    main()
