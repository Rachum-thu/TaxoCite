import argparse
from langchain_openai import ChatOpenAI

from data.raw.util.prompts import citation_extraction_prompt

def main():
    parser = argparse.ArgumentParser(description="Extract references from text files using LLM (batch mode)")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input text files")
    parser.add_argument("--outputs", nargs='+', required=True, help="List of output YAML files")
    args = parser.parse_args()

    # Validate input/output count match
    if len(args.inputs) != len(args.outputs):
        print(f"Error: Number of inputs ({len(args.inputs)}) must match number of outputs ({len(args.outputs)})")
        exit(1)

    # Read all input texts
    print(f"Reading {len(args.inputs)} input files...")
    reference_texts = []
    for input_path in args.inputs:
        with open(input_path, "r", encoding="utf-8") as f:
            reference_texts.append(f.read())
        print(f"  Read: {input_path}")

    # Prepare prompts
    prompts = [citation_extraction_prompt.format(reference_text=text) for text in reference_texts]

    # Batch extract using LLM
    print(f"\nProcessing {len(prompts)} files in batch (max_concurrency=16)...")
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0, reasoning_effort="medium")
    outputs = llm.batch(prompts, config={"max_concurrency": 16})

    # Write outputs
    print("\nWriting output files...")
    for i, (output_path, output) in enumerate(zip(args.outputs, outputs)):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output.content)
        print(f"  Written: {output_path}")

    print(f"\nSuccessfully extracted references from {len(args.inputs)} files!")


if __name__ == "__main__":
    main()
