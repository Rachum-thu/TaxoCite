import argparse
import os
from pathlib import Path
from langchain_openai import ChatOpenAI

# from data.raw.util.prompts import taxonomy_induction_prompt as taxonomy_prompt
from data.raw.util.prompts import taxonomy_induction_prompt_paper_and_survey as taxonomy_prompt


def main():
    parser = argparse.ArgumentParser(description="Build topic taxonomy from survey markdown files")
    parser.add_argument("--survey_dir", required=True, help="Directory containing survey markdown files")
    parser.add_argument("--paper_dir", required=False, default=None, help="Directory containing paper markdown files")
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
        with open(md_file, "r", encoding="utf-8") as f:
            survey_texts.append(f.read())

    combined_survey_text = "\n\n(next survey)\n\n".join(survey_texts)
    print(f"\nCombined survey text length: {len(combined_survey_text)} characters")
    
    # Check if paper directory is provided
    if args.paper_dir is None:
        print("No paper directory provided, skipping paper content")
        paper_texts = []
    else:
        # Read all markdown files in the directory
        paper_dir = Path(args.paper_dir)
        if not paper_dir.exists() or not paper_dir.is_dir():
            print(f"Error: Paper directory '{args.paper_dir}' does not exist or is not a directory")
            exit(1)
        
        # Read all markdown files in the directory
        md_files = sorted(paper_dir.glob("*.md"))
        if not md_files:
            print(f"Error: No markdown files found in '{args.paper_dir}'")
            exit(1)
        
        print(f"Found {len(md_files)} paper markdown files:")
        paper_texts = []
        for md_file in md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                paper_texts.append(f.read())

        combined_paper_text = "\n\n(next paper)\n\n".join(paper_texts)
        print(f"\nCombined paper text length: {len(combined_paper_text)} characters")
    # Prepare prompt
    prompt = taxonomy_prompt.format(topic_name=args.topic_name, paper_text=combined_paper_text if combined_paper_text else "No paper content found", survey_text=combined_survey_text)

    # Generate taxonomy using LLM
    print(f"\nGenerating taxonomy for topic: {args.topic_name}")
    llm = ChatOpenAI(model="gpt-5-2025-08-07", temperature=0, reasoning_effort="medium")
    output = []
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
        output.append(chunk.content)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(output))

    print(f"\n\nTaxonomy successfully written to: {args.output}")


if __name__ == "__main__":
    main()
