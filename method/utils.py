import yaml
import re


class QuotedDumper(yaml.SafeDumper):
    pass


QuotedDumper.add_representer(str, lambda d, data: d.represent_scalar('tag:yaml.org,2002:str', data, style='"'))


def extract_blocks(segmented_markdown):
    blocks = re.findall(r'<block id="\d+">\s*(.*?)\s*</block>', segmented_markdown, re.DOTALL)
    return blocks


def extract_taxonomy_nodes(taxonomy_text):
    """Extract all node titles from markdown taxonomy."""
    nodes = []
    for line in taxonomy_text.split('\n'):
        line = line.strip()
        if line.startswith('#'):
            # Remove leading # symbols and clean up
            title = re.sub(r'^#+\s*', '', line).strip()
            if title:
                nodes.append(title)
    return nodes


def normalize_label(label):
    """Normalize label for matching."""
    if not label:
        return ""
    # Convert to lowercase, remove extra whitespace
    normalized = re.sub(r'\s+', ' ', label.lower().strip())
    return normalized


def validate_label(label, valid_nodes, default="Other"):
    """Validate label against taxonomy nodes."""
    if not label:
        return default

    normalized_input = normalize_label(label)

    # Build normalized node mapping
    for node in valid_nodes:
        if normalize_label(node) == normalized_input:
            return node

    return default


def mask_segmented_markdown(segmented_markdown, target_block_id):
    """
    Mask all blocks except the target block.
    Keep all section headers, replace other block contents with [...].
    """
    lines = segmented_markdown.split('\n')
    result = []
    current_block_id = None
    in_block = False

    for line in lines:
        # Check for block start
        if '<block id="' in line:
            match = re.search(r'<block id="(\d+)">', line)
            if match:
                current_block_id = int(match.group(1))
                in_block = True
                if current_block_id != target_block_id:
                    result.append('[...]')
                continue

        # Check for block end
        if '</block>' in line:
            in_block = False
            current_block_id = None
            continue

        # If we're in the target block, keep the line
        if in_block and current_block_id == target_block_id:
            result.append(line)
        # If we're not in a block (headers, etc), keep the line
        elif not in_block:
            result.append(line)

    return '\n'.join(result)
