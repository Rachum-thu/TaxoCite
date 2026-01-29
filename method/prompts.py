zeroshot_intent_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to classify the citation intent for each marker listed in `Available markers`, using the provided taxonomy.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Taxonomy (a tree of label names with definitions):
{taxonomy}

Task rules:
1) Produce EXACTLY ONE label for EACH marker (one label per marker).
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Choose the MOST SPECIFIC matching taxonomy NODE for the marker, based on how the cited work is used/described in the current block.

Label format contract (critical):
- Your output label MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim from the taxonomy).
- Output ONLY the node name itself, NOT a path. 

Disambiguation rules:
A) Benchmarks: citations to benchmarks/datasets used for reporting results -> "Benchmark Utilization".
B) Evaluation metrics/judges/scoring rules -> "Metrics Utilization".
C) Baselines/SOTA/prior numbers used for comparison -> "Result Comparison".
D) Evaluation procedure/splits/pipelines/prompting used for evaluation -> "Setting/Protocal Adoption".
E) Resources used to build/run the method (datasets as training data, tools/libraries/KBs/code used for producing outputs) -> "Resource Utilization".
F) Adoption of model/architecture -> "Model/Architecture Adoption".
G) Adoption of algorithmic procedure/objective/training trick/inference/optimization method -> "Algorithm/Principle Adoption".
H) Pure framing/survey/motivation without adopting as method/eval -> pick the most specific "Background" child that fits.
I) If none of the taxonomy nodes apply -> "Other Intent".

Return the label for each marker.
"""


zeroshot_topic_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to classify the TOPIC for each citation marker listed in `Available markers`, using the provided topic taxonomy.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Topic taxonomy (a tree of topic label names with definitions):
{taxonomy}

Task rules:
1) Produce EXACTLY ONE topic label for EACH marker (one label per marker).
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Choose the MOST SPECIFIC matching taxonomy NODE for the marker, based on what the cited work is about as described/implied in the current block (and supported by the abstract if needed).

Label format contract (critical):
- Your output label MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim from the taxonomy).
- Output ONLY the node name itself, NOT a path.

Return the topic label for each marker.
"""

cot_intent_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to classify the citation intent for each marker listed in `Available markers`, using the provided taxonomy.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Taxonomy (a tree of label names with definitions):
{taxonomy}

Task rules:
1) Produce EXACTLY ONE record for EACH marker.
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Choose the MOST SPECIFIC matching taxonomy NODE for the marker, based on how the cited work is used/described in the current block.

Output fields (critical):
- analysis: ONE sentence (<= 20 words) summarizing the marker's role/evidence in this block. Do NOT write step-by-step reasoning.
- label: MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim).
  - Output ONLY the node name itself, NOT a path.
  - Valid: "Algorithm/Principle Adoption"
  - Invalid: "Method/Algorithm/Principle Adoption"
  - No extra text.

Disambiguation rules:
A) Benchmarks used for reporting results -> "Benchmark Utilization".
B) Metrics/judges/scoring rules -> "Metrics Utilization".
C) Baselines/SOTA/prior numbers used for comparison -> "Result Comparison".
D) Evaluation procedure/splits/pipelines/prompting used for evaluation -> "Setting/Protocal Adoption".
E) Resources used to build/run the method -> "Resource Utilization".
F) Adoption of model/architecture -> "Model/Architecture Adoption".
G) Adoption of algorithmic procedure/objective/training trick/inference/optimization method -> "Algorithm/Principle Adoption".
H) Pure framing/survey/motivation without adopting as method/eval -> choose the most specific "Background" child.
I) If none apply -> "Other Intent".

Return the classification for each marker.
"""


cot_topic_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to classify the TOPIC for each citation marker listed in `Available markers`, using the provided topic taxonomy.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Topic taxonomy (a tree of topic label names with definitions):
{taxonomy}

Task rules:
1) Produce EXACTLY ONE record for EACH marker.
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Choose the MOST SPECIFIC matching taxonomy NODE for the marker, based on what the cited work is about as described/implied in the current block.

Output fields (critical):
- analysis: ONE sentence (<= 20 words) summarizing the key topic evidence near the marker. Do NOT write step-by-step reasoning.
- label: MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim).
  - Output ONLY the node name itself, NOT a path.
  - No extra text.

Topic evidence hints:
- Prefer explicit cues near the marker: "X, a Y benchmark", method/dataset/task names, short descriptions.
- If multiple mentions differ, choose the dominant topic for this block.

Return the topic label for each marker.
"""


reflexion_intent_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to REVIEW and REFINE citation intent classifications made by a previous annotator.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Taxonomy (a tree of label names with definitions):
{taxonomy}

- Previous annotations (mapping from marker -> ONE node name; may be wrong):
{previous_annotations}

Task:
1) For EACH marker, review the previous annotation and decide whether to AGREE or REVISE. Previous annotations are fallible; prioritize the visible evidence and taxonomy definitions.
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Produce:
   - critique: exactly ONE sentence explaining why you keep or change the previous label.
   - label: your FINAL label (may be the same or different).

Label format contract (critical):
- label MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim from the taxonomy).
- Output ONLY the node name itself, NOT a path (e.g., never "Method/Algorithm/Principle Adoption").

Return the critique and final label for each marker.
"""

reflexion_topic_prompt = """You are given (1) the paper abstract and (2) the paper structure where ONLY the current block is visible and all other blocks are masked as "[...]". Your job is to REVIEW and REFINE topic classifications made by a previous annotator.

Inputs:
- Abstract:
{abstract}

- Paper structure (current block shown; other blocks masked):
{masked_markdown}

- Available markers (guaranteed to appear in the current block):
{markers}

- Topic taxonomy (a tree of topic label names with definitions):
{taxonomy}

- Previous annotations (mapping from marker -> ONE node name; may be wrong):
{previous_annotations}

Task:
1) For EACH marker, review the previous annotation and decide whether to AGREE or REVISE. Previous annotations are fallible; prioritize the visible evidence and taxonomy definitions.
2) Use evidence from: (a) the current block, and (b) the abstract. Ignore masked parts.
3) Produce:
   - critique: exactly ONE sentence explaining why you keep or change the previous label.
   - label: your FINAL topic label (may be the same or different).

Label format contract (critical):
- label MUST be the EXACT NAME of ONE taxonomy node (copy the node title verbatim from the taxonomy).
- Output ONLY the node name itself, NOT a path.
- Do NOT add extra text.

Return the critique and final topic label for each marker.
"""
