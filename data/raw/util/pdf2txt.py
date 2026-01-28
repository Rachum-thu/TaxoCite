import argparse
from langchain_community.document_loaders import PyPDFLoader


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to text file")
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument("output_txt", help="Path to output text file")
    args = parser.parse_args()

    # Load PDF
    loader = PyPDFLoader(args.input_pdf)
    pages = loader.load()

    # Extract text
    text = "\n".join([page.page_content for page in pages])

    # Write to output file
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Successfully converted {args.input_pdf} to {args.output_txt}")


if __name__ == "__main__":
    main()
