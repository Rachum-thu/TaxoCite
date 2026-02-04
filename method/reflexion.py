import argparse
import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from method.prompts import zeroshot_intent_prompt, zeroshot_topic_prompt, reflexion_intent_prompt, reflexion_topic_prompt
from method.utils import QuotedDumper, extract_blocks, extract_taxonomy_nodes, validate_label, mask_segmented_markdown


class Citation(BaseModel):
    marker: str
    label: str


class CitationClassification(BaseModel):
    classifications: list[Citation]


class ReflexionCitation(BaseModel):
    marker: str
    critique: str
    label: str


class ReflexionClassification(BaseModel):
    classifications: list[ReflexionCitation]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_intent_taxonomy", required=True)
    parser.add_argument("--domain_topic_taxonomy", required=True)
    parser.add_argument("--seg_paper", required=True)
    parser.add_argument("--ref_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--zeroshot_result", default=None, help="Optional: reuse zeroshot results instead of running Round 1")
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

        intent_prompts.append(zeroshot_intent_prompt.format(
            markers=', '.join(markers),
            taxonomy=intent_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md
        ))

        topic_prompts.append(zeroshot_topic_prompt.format(
            markers=', '.join(markers),
            taxonomy=topic_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md
        ))

        block_ids_to_process.append(block_id)

    # Round 1: Zero-shot classification
    llm = ChatOpenAI(model=args.model, reasoning_effort="medium")

    if args.zeroshot_result:
        # Reuse existing zeroshot results
        print(f"Round 1: Loading zeroshot results from {args.zeroshot_result}...")
        zeroshot_data = yaml.safe_load(open(args.zeroshot_result))
        zeroshot_citations = {c.get("unique_context_marker", ""): c for c in zeroshot_data.get("citations", [])}

        # Rebuild intent_outputs_r1 and topic_outputs_r1 from zeroshot results
        intent_outputs_r1 = []
        topic_outputs_r1 = []

        for block_id in block_ids_to_process:
            block_citations = block_to_citations[block_id]

            # Build CitationClassification for intent
            intent_classifications = []
            topic_classifications = []

            for bc in block_citations:
                marker = bc["marker"]
                idx = bc["idx"]
                zeroshot_cit = zeroshot_citations.get(marker, {})

                intent_label = zeroshot_cit.get("intent_labels", [])[idx] if idx < len(zeroshot_cit.get("intent_labels", [])) else "Other Intent"
                topic_label = zeroshot_cit.get("topic_labels", [])[idx] if idx < len(zeroshot_cit.get("topic_labels", [])) else "Other Topic"

                intent_classifications.append(Citation(marker=marker, label=intent_label))
                topic_classifications.append(Citation(marker=marker, label=topic_label))

            intent_outputs_r1.append(CitationClassification(classifications=intent_classifications))
            topic_outputs_r1.append(CitationClassification(classifications=topic_classifications))
    else:
        # Run zeroshot classification from scratch (intent and topic in parallel)
        from concurrent.futures import ThreadPoolExecutor
        print("Round 1: Zero-shot classification...")
        zeroshot_llm = llm.with_structured_output(CitationClassification)
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(zeroshot_llm.batch, intent_prompts, {"max_concurrency": 32})
            topic_future = executor.submit(zeroshot_llm.batch, topic_prompts, {"max_concurrency": 32})
            intent_outputs_r1 = intent_future.result()
            topic_outputs_r1 = topic_future.result()

    # Round 2: Reflexion - critique and refine
    print("Round 2: Reflexion...")
    reflexion_intent_prompts = []
    reflexion_topic_prompts = []

    for block_id, intent_output_r1, topic_output_r1 in zip(block_ids_to_process, intent_outputs_r1, topic_outputs_r1):
        block_citations = block_to_citations[block_id]
        markers = [bc["marker"] for bc in block_citations]
        masked_md = mask_segmented_markdown(segmented_markdown, block_id)

        # Build previous annotation strings
        intent_prev = {c.marker: c.label for c in intent_output_r1.classifications}
        topic_prev = {c.marker: c.label for c in topic_output_r1.classifications}

        prev_intent_str = "\n".join([f"{m} -> {intent_prev.get(m, 'N/A')}" for m in markers])
        prev_topic_str = "\n".join([f"{m} -> {topic_prev.get(m, 'N/A')}" for m in markers])

        reflexion_intent_prompts.append(reflexion_intent_prompt.format(
            markers=', '.join(markers),
            taxonomy=intent_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md,
            previous_annotations=prev_intent_str
        ))

        reflexion_topic_prompts.append(reflexion_topic_prompt.format(
            markers=', '.join(markers),
            taxonomy=topic_taxonomy,
            abstract=abstract,
            masked_markdown=masked_md,
            previous_annotations=prev_topic_str
        ))

    # Round 2 batch (intent and topic in parallel)
    from concurrent.futures import ThreadPoolExecutor
    reflexion_llm = llm.with_structured_output(ReflexionClassification)
    with ThreadPoolExecutor(max_workers=2) as executor:
        intent_future = executor.submit(reflexion_llm.batch, reflexion_intent_prompts, {"max_concurrency": 32})
        topic_future = executor.submit(reflexion_llm.batch, reflexion_topic_prompts, {"max_concurrency": 32})
        intent_outputs_r2 = intent_future.result()
        topic_outputs_r2 = topic_future.result()

    # Fill results with validation
    for block_id, intent_output, topic_output in zip(block_ids_to_process, intent_outputs_r2, topic_outputs_r2):
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
