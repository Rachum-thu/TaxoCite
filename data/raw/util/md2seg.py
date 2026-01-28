import argparse
import yaml
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from data.raw.util.prompts import extract_title_abstract_prompt

class PaperMetadata(BaseModel):
    title: str = Field(description="The title of the paper")
    abstract: str = Field(description="The abstract of the paper. If no abstract exists, provide a concise 2-3 sentence summary.")
    abstract_is_verbatim: bool = Field(description="True if the abstract was explicitly present and extracted verbatim; False if it was summarized.")

def segment_markdown(md_content):
    """Parse markdown and add <block> tags around content sections."""
    lines = md_content.split('\n')
    result = []
    block_id = 0
    in_block = False

    for line in lines:
        # Check if this is a ## or ### header (block boundary)
        if line.startswith('### ') or line.startswith('## '):
            # Close previous block if open
            if in_block:
                result.append('</block>')
                in_block = False

            # Add the header line
            result.append(line)

            # Start new block
            result.append(f'<block id="{block_id}">')
            block_id += 1
            in_block = True
        else:
            # Regular content line
            result.append(line)

    # Close final block if still open
    if in_block:
        result.append('</block>')

    return '\n'.join(result)

def main():
    parser = argparse.ArgumentParser(description="Parse markdown papers into structured segments")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input markdown files")
    parser.add_argument("--outputs", nargs='+', required=True, help="List of output YAML files")
    args = parser.parse_args()

    # Read all markdown files
    md_contents = []
    for input_path in args.inputs:
        with open(input_path, "r") as f:
            md_contents.append(f.read())

    # Prepare prompts for title/abstract extraction
    prompts = [extract_title_abstract_prompt.format(md_content=content) for content in md_contents]

    # Batch process with LLM using structured output
    llm = ChatOpenAI(model="gpt-5-mini-2025-08-07", temperature=0)
    structured_llm = llm.with_structured_output(PaperMetadata)
    outputs = structured_llm.batch(prompts, config={"max_concurrency": 64})

    # Write outputs
    for input_path, output_path, md_content, output in zip(args.inputs, args.outputs, md_contents, outputs):
        segmented = segment_markdown(md_content)
        result = {
            "title": output.title,
            "abstract": output.abstract,
            "abstract_is_verbatim": output.abstract_is_verbatim,
            "segmented_markdown": segmented
        }
        with open(output_path, "w") as f:
            yaml.dump(result, f, allow_unicode=True, sort_keys=False)

if __name__ == "__main__":
    main()
