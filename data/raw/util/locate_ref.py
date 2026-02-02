import argparse
import yaml
import re
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class QuotedDumper(yaml.SafeDumper):
    pass


QuotedDumper.add_representer(str, lambda d, data: d.represent_scalar('tag:yaml.org,2002:str', data, style='"'))


class BlockCitations(BaseModel):
    markers: list[str]


class ExtractedMarkers(BaseModel):
    markers: list[str] = Field(
        description="List of citation markers found in the text. Examples: ['[1]', '[2]'], ['(Smith, 2020)', '(Jones et al., 2019)'], ['^1', '^2'], etc."
    )


def extract_blocks(segmented_markdown):
    blocks = re.findall(r'<block id="\d+">\s*(.*?)\s*</block>', segmented_markdown, re.DOTALL)
    assert len(blocks) == segmented_markdown.count('</block>'), "Block format error"
    return blocks


def extract_citations_from_text(text):
    citation_numbers = set()
    for match in re.finditer(r'\[([^\]]+)\]', text):
        for part in match.group(1).split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    citation_numbers.update(range(start, end + 1))
                except:
                    pass
            else:
                try:
                    citation_numbers.add(int(part))
                except:
                    pass
    return citation_numbers


def locate_refs_acm(blocks, citations):
    citation_map = {}
    for c in citations:
        c["block_ids"] = []
        match = re.search(r'\[(\d+)\]', c.get("unique_context_marker", ""))
        if match:
            citation_map[int(match.group(1))] = c

    for block_id, block in enumerate(blocks):
        for num in extract_citations_from_text(block):
            if num in citation_map:
                citation_map[num]["block_ids"].append(block_id)


def locate_refs_llm(blocks, citations):
    marker_map = {c.get("unique_context_marker", ""): c for c in citations if c.get("unique_context_marker")}
    for c in citations:
        c["block_ids"] = []

    if not marker_map:
        return

    prompts = []
    for block in blocks:
        prompt = f"""Given the following block of text from an academic paper, identify which citation markers appear in this block.

Available citation markers: {', '.join(marker_map.keys())}

Block text:
{block}

Output only the citation markers that actually appear in this block."""
        prompts.append(prompt)

    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", reasoning_effort="medium").with_structured_output(BlockCitations)
    outputs = llm.batch(prompts, config={"max_concurrency": 32})

    for block_id, output in enumerate(outputs):
        for marker in output.markers:
            if marker in marker_map:
                marker_map[marker]["block_ids"].append(block_id)


def locate_refs_no_meta(blocks, citations):
    """
    Extract citation markers from blocks without pre-existing metadata.
    Discovers markers automatically and builds citation list from scratch.
    """
    # Prepare prompts to extract markers from each block
    prompts = []
    for block in blocks:
        prompt = f"""Given the following block of text from an academic paper, extract ALL citation markers that appear in this block.

Citation markers can be in various formats such as:
- Numbered brackets: [1], [2], [3]
- Author-year: (Smith, 2020), (Jones et al., 2019)
- Superscript numbers: ^1, ^2
- Parenthetical numbers: (1), (2)
- Or other citation formats

Extract ONLY the citation markers (references to other works), not other bracketed content like [TODO], [Figure 1], etc.

Block text:
{block}

Output all citation markers found in this block exactly as they appear."""
        prompts.append(prompt)

    # Batch process with LLM
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", reasoning_effort="medium").with_structured_output(ExtractedMarkers)
    outputs = llm.batch(prompts, config={"max_concurrency": 32})

    # Dynamically build marker_map
    marker_map = {}
    for block_id, output in enumerate(outputs):
        for marker in output.markers:
            if marker not in marker_map:
                marker_map[marker] = {
                    "unique_context_marker": marker,
                    "block_ids": []
                }
            marker_map[marker]["block_ids"].append(block_id)

    # Deduplicate block_ids for each marker
    for marker_data in marker_map.values():
        marker_data["block_ids"] = sorted(set(marker_data["block_ids"]))

    # Update citations list
    citations.clear()
    citations.extend(marker_map.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_papers", nargs='+', required=True)
    parser.add_argument("--yaml_refs", nargs='+', required=True)
    parser.add_argument("--mode", choices=["llm", "acm", "no_meta"], required=True)
    args = parser.parse_args()

    assert len(args.seg_papers) == len(args.yaml_refs), "Mismatched input counts"

    print(f"Mode: {args.mode}")
    print(f"Processing {len(args.seg_papers)} files...")

    for seg_path, ref_path in zip(args.seg_papers, args.yaml_refs):
        print(f"  Processing: {seg_path}")

        seg_data = yaml.safe_load(open(seg_path))

        # For no_meta mode, start with empty citations
        if args.mode == "no_meta":
            citations = []
        else:
            ref_data = yaml.safe_load(open(ref_path))
            citations = ref_data.get("citations", [])

        blocks = extract_blocks(seg_data.get("segmented_markdown", ""))

        print(f"    Found {len(blocks)} blocks")

        # Select appropriate function based on mode
        if args.mode == "acm":
            locate_refs_acm(blocks, citations)
        elif args.mode == "llm":
            locate_refs_llm(blocks, citations)
        elif args.mode == "no_meta":
            locate_refs_no_meta(blocks, citations)

        print(f"    Extracted {len(citations)} citations")

        with open(ref_path, "w") as f:
            yaml.dump({"citations": citations}, f, Dumper=QuotedDumper, allow_unicode=True, sort_keys=False, width=1000000)

        print(f"    Written to: {ref_path}")

if __name__ == "__main__":
    main()
