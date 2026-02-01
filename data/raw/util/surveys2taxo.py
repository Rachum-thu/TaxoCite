import argparse
import os
from pathlib import Path
from langchain_openai import ChatOpenAI

from data.raw.util.prompts import taxonomy_induction_prompt as taxonomy_prompt


def main():
    parser = argparse.ArgumentParser(description="Build topic taxonomy from survey markdown files")
    parser.add_argument("--survey_dir", required=True, help="Directory containing survey markdown files")
    parser.add_argument("--topic_name", required=True, help="Topic name for the taxonomy")
    parser.add_argument("--output", required=True, help="Output taxonomy markdown file path")
    args = parser.parse_args()

    # Check if survey directory exists
    survey_dir = Path(args.survey_dir)
    if not survey_dir.exists() or not survey_dir.is_dir():
        print(f"Error: Survey directory '{args.survey_dir}' does not exist or is not a directory")
        exit(1)

    # Read all markdown files in the directory
    md_files = sorted(survey_dir.glob("*.md"))
    if not md_files:
        print(f"Error: No markdown files found in '{args.survey_dir}'")
        exit(1)

    print(f"Found {len(md_files)} survey markdown files:")
    survey_texts = []
    for md_file in md_files:
        print(f"  Reading: {md_file.name}")
        with open(md_file, "r", encoding="utf-8") as f:
            survey_texts.append(f.read())

    # Concatenate all survey texts with separator
    combined_survey_text = "\n\n(next survey)\n\n".join(survey_texts)
    print(f"\nCombined survey text length: {len(combined_survey_text)} characters")

    # Prepare prompt
    prompt = taxonomy_prompt.format(topic_name=args.topic_name, survey_text=combined_survey_text)

    # Generate taxonomy using LLM
    print(f"\nGenerating taxonomy for topic: {args.topic_name}")
    llm = ChatOpenAI(model="gpt-5-2025-08-07", temperature=0, reasoning_effort="medium")
    output = []
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        output.append(chunk.content)

    # Post-process: add "Other Topics" category for fault tolerance
    taxonomy_text = "".join(output).strip()

    # Check if "Other" category already exists at level 2
    if not any(line.strip().startswith("## Other") for line in taxonomy_text.split("\n")):
        # Add "Other Topics" category at the end
        other_category = "\n\n## Other Topics\n**Description:** Topics and methods that do not fit into the above categories or are emerging areas."
        taxonomy_text += other_category
        print("\n\n[Auto-added: ## Other Topics for fault tolerance]")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(taxonomy_text)

    print(f"\n\nTaxonomy successfully written to: {args.output}")


if __name__ == "__main__":
    main()
