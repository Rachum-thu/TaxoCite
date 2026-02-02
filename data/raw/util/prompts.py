taxonomy_induction_prompt_paper_and_survey = '''
System Role:
You are a Senior Domain Scientist and Taxonomist. Your goal is to construct a reusable taxonomy that captures how prior work (references) is conceptually used across papers within the same research topic.

User Prompt:
**Task:** Construct a topic-specific taxonomy for the research topic: "{topic_name}".

**Input Data:**
* **Topic Name:** {topic_name}
* **Paper and Survey Content:** (Provided below)
* Note: Input content may contain multiple papers and surveys; synthesize a single unified taxonomy by jointly considering all provided contents.

**Core Requirement:**
Infer a single, unified taxonomy that captures how references are conceptually used across non-survey research papers within the provided topic. 

1. The Paper Content provides examples of how references can be conceptually used. Survey Content aids to (a) clarify terminology and (b) consolidate overlapping concepts, and (c) introduce additional conceptual terms that are *parallel to* those observed in the Paper Content.
2. The taxonomy MUST reflect topic-specific reference functions rather than generic paper sections (e.g., avoid labels such as "Introduction", "Related Work", "Methods"). When evidence is weak or inconsistent, prefer higher-level, safer abstractions. 
3. The taxonomy MUST be reusable across multiple research papers under the provided topic beyond the provided Paper Content. Avoid specific paper titles, benchmark names, dataset names, tool names, or model names.
4. The topic may be interdisciplinary. When reference usage reflects different disciplinary perspectives, the taxonomy should explicitly represent these lenses, rather than collapsing them into a single-domain framing. Identify the primary disciplinary perspectives that dominate reference usage, and represent secondary or supporting disciplines at a coarser level of granularity.

**Instructions:**
1. **Analyze the Paper and Survey Content:** Identify the intent of the references in the Paper Content, including how they are used to frame problems, motivate methods, support assumptions, or interpret results. Survey Content is used only as secondary support to clarify terminology, merge overlapping concepts, or introduce additional *parallel* conceptual terms that fit the Paper Content. 
2. **Extract the Hierarchy:** Build a taxonomy where:
   - **Level 1 (#):** Must be exactly "{topic_name}".
   - **Level 2 (##):** Major dimensions/paradigms of the topic. 
   - **Level 3 (###):** Specific sub-categories or methodological classes.
   - Hard Constraint: Do NOT use level-4 headers (####). The deepest allowed level is ###.
3. **Level-2 Identification (Logic-First):** Identify all Level-2 nodes first, before expanding the hierarchy. Each Level-2 node must represent a distinct conceptual role in how references are used, with no overlap between nodes. Together, the Level-2 nodes must be sufficient to cover the major reference-usage logic under "{topic_name}".
4. **Granularity Control (Crucial):**
   - **Total Node Count:** 20-30 nodes total (including the root).
   - **Depth (Strict):** Exactly 3 levels total (#, ##, ###). Do not produce any #### nodes.
   - **Leaf Nodes:** Must be concepts/method categories (e.g., "Location Encoder"), NOT specific paper titles, benchmark names, dataset names, tool names, or model names.
   -**Exclusivity Requirement:** Leaf nodes should be conceptually distinct and non-overlapping across the taxonomy. Avoid duplicating the same underlying concept under different disciplinary perspectives.
5. **Descriptions:**
   - Every node (except for the root) MUST have a one-line description immediately under it.
   - Use this exact format for the description line:
     `**Description:** <one concise sentence (8-20 words) summarizing this node>`
   - Descriptions must be neutral, factual, and based on the Paper and Survey Content (no citations).
   - Do not repeat the node name verbatim in its description.
6. **Formatting (Strict):**
   - Output ONLY the taxonomy.
   - Use Markdown headers for nodes (#, ##, ###).
   - Under each header, include exactly ONE description line as specified.
   - No extra bullet points, no numbering, no citations, no commentary, no blank sections.
   
An example taxonomy for the topic "retrieval augmented generation" is:
"""
# retrieval augmented generation

## Retrieval
**Description:** Design choices for accessing external knowledge, balancing relevance, efficiency, and adaptability.

### Retriever Type
**Description:** Sparse keyword matching versus trainable dense encoders for semantic similarity retrieval.

### Retrieval Granularity
**Description:** Selection of indexing units such as documents, chunks, tokens, entities, or mentions.

### Indexing and Query Optimization
**Description:** Chunking, metadata, hierarchical indices, and query rewriting/expansion to improve recall and precision.
...
"""

**Paper Content:**
"""
Paper Content:
{paper_text}

Survey Content:
{survey_text}
"""

**Output:**
'''

taxonomy_induction_prompt = '''
System Role:
You are a Senior Domain Scientist and Taxonomist. Your goal is to structure complex scientific topics into clear, hierarchical taxonomies grounded in the provided survey text.

User Prompt:
**Task:** Construct a domain-specific topic taxonomy for the research topic: "{topic_name}".

**Input Data:**
* **Topic Name:** {topic_name}
* **Survey Paper Content:** (Provided below)
* Note: survey text may contain multiple survey papers; synthesize a single unified taxonomy by jointly considering all provided survey contents, and include a category only if it is supported by at least one survey.

**Core Requirement:**
The taxonomy MUST be derived from the survey text. Be conservative: do not introduce categories that are not supported by the survey. If the survey is unclear, choose higher-level, safer abstractions.

**Instructions:**
1. **Analyze the Survey:** Identify the classification scheme used by the authors (often in "Introduction", "Related Work", or a "Taxonomy" section). Prefer the authors' own dimensions and terminology when available.
2. **Extract the Hierarchy:** Build a taxonomy where:
   - **Level 1 (#):** Must be exactly "{topic_name}".
   - **Level 2 (##):** Major dimensions/paradigms of the topic.
   - **Level 3 (###):** Specific sub-categories or methodological classes.
   - Hard Constraint: Do NOT use level-4 headers (####). The deepest allowed level is ###.
3. **Granularity Control (Crucial):**
   - **Total Node Count:** 20-30 nodes total (including the root).
   - **Depth (Strict):** Exactly 3 levels total (#, ##, ###). Do not produce any #### nodes.
   - **Leaf Nodes:** Must be concepts/method categories (e.g., "Contrastive Learning"), NOT specific paper titles, benchmark names, dataset names, tool names, or model names.
4. **Descriptions:**
   - Every node (except for the root) MUST have a one-line description immediately under it.
   - Use this exact format for the description line:
     `**Description:** <one concise sentence (8-20 words) summarizing this node>`
   - Descriptions must be neutral, factual, and based on survey content (no citations).
   - Do not repeat the node name verbatim in its description.
5. **Formatting (Strict):**
   - Output ONLY the taxonomy.
   - Use Markdown headers for nodes (#, ##, ###).
   - Under each header, include exactly ONE description line as specified.
   - No extra bullet points, no numbering, no citations, no commentary, no blank sections.

**Survey Paper Content:**
"""
{survey_text}
"""

**Output:**
'''

ocr_modify_prompt = """
System:
You are an OCR(noisy text)-to-Markdown extractor for academic papers. Extract ONLY the main-body text from the OCR input and output valid Markdown. Be conservative: do NOT summarize, paraphrase, reorder, or invent. Output only text that appears in the OCR.

Include:
- Title
- Abstract
- Main-body sections and subsections (e.g., Introduction, Related Work, Method, Experiments, Discussion, Conclusion, Limitations/Ethics if present)

Exclude (remove completely):
- References/Bibliography
- Appendix/Appendices/Supplementary/Additional Results/Proofs
- Figures/tables and their captions (e.g., lines starting with "Figure", "Fig.", "Table" + caption text)
- Footnotes, acknowledgements, author notes, funding, boilerplate/copyright, page headers/footers, page numbers

Markdown rules:
- Exactly one H1 for the title.
- Use H2 for top-level sections; H3 for subsections only if clearly indicated.
- Merge broken line wraps into paragraphs; fix obvious hyphenation at line breaks (e.g., "inter-\nnational" -> "international").
- Preserve in-text citations exactly (e.g., "[1]", "[2,3]", "(Smith et al., 2020)").
- Preserve inline math as text; do not include displayed equations as figures.
- No images/tables. No commentary. Output Markdown only.

OCR input:
<<<OCR>>>
{ocr_text}
<<<END OCR>>>
"""

citation_extraction_prompt = '''
System Instructions:
You are an expert bibliographer and reference-list parser.

TASK SCOPE (READ CAREFULLY):
You are NOT extracting metadata for the current paper/document.
You are extracting the titles of works that THIS paper cites, i.e., items in its Reference List / Bibliography.

INPUT GUARANTEE:
The user will provide OCR text that corresponds to the References/Bibliography section (or a superset that contains it).
The text may contain OCR noise tags (e.g., ``), broken lines, and formatting artifacts.

YOUR GOAL:
1. Identify the Reference List / Bibliography entries (the cited works), typically marked by patterns such as:
   - [1], [2], ...
   - 1., 2., ...
   - (1), (2), ...
   - or other consistent reference-item markers
2. Clean OCR noise and merge broken lines inside each entry.
3. For EACH reference entry, extract ONLY the Paper/Work Title.

CRITICAL DEFINITION - "TITLE":
The `title` field is ONLY the specific name of the cited work.
- EXCLUDE Authors: do not include any person/organization names (e.g., "Josh Achiam", "Meta AI", "et al.").
- EXCLUDE Venue/Publisher: do not include "arXiv preprint", "Proceedings of...", "NeurIPS", "OpenAI Blog", journal/conference names.
- EXCLUDE Year: do not include publication year.
- EXCLUDE URLs/DOIs: do not include "https://..." or "doi:...".
- EXCLUDE the current paper’s own title or metadata (this is not a self-metadata extraction task).

EXAMPLES OF PARSING:
- Input entry: "[1] Brown, T., et al. 2020. Language models are few-shot learners. NeurIPS."
  - Output title: "Language models are few-shot learners"
- Input entry: "[2] Meta AI. Meta LLaMA 3. https://ai.meta.com"
  - Output title: "Meta LLaMA 3"

Output Format (YAML only):
Strictly output valid YAML and NOTHING else.

citations:
  - title: "<ONLY the cited work's title>"
    unique_context_marker: "<the item marker exactly as seen, e.g., [1] or 1.>"

User Input (REFERENCES SECTION OCR TEXT ONLY):
<REFERENCES_OCR_TEXT>
{reference_text}
</REFERENCES_OCR_TEXT>
'''

extract_title_abstract_prompt = """
You are given the full Markdown of an academic paper.

Task: return (1) the paper title and (2) the abstract.

TITLE extraction rules (in priority order):
1) If the Markdown begins with a single H1 heading (a line starting with "# "), use that as the title.
2) Else, if there is a YAML front-matter field named "title", use it.
3) Else, use the first non-empty line that looks like a title (short, not a heading like "Contents", not an author list, not a section header such as "Introduction").

ABSTRACT extraction rules:
A) If an explicit abstract exists, extract it VERBATIM, preserving line breaks as reasonable, without adding or removing content.
   Consider the abstract explicit if it appears under:
   - a heading like "Abstract", "ABSTRACT" (any level of Markdown heading), OR
   - a bold label like "**Abstract**" followed by text.
   The abstract content ends before the next top-level section heading (e.g., "Introduction", "1 Introduction", "Related Work", etc.) or before another heading of the same or higher level.
B) If no explicit abstract exists, write a 2–3 sentence summary based only on the paper’s introduction and main contributions described in the Markdown.
   Do NOT invent results, numbers, datasets, or claims not supported by the text.
   Avoid citations/URLs and avoid phrases like "This paper proposes" repeated; be specific about (i) problem, (ii) method, (iii) main findings/claims (only if stated).

Return only the fields required by the schema.

Paper markdown:
<<<BEGIN_PAPER>>>
{md_content}
<<<END_PAPER>>>
"""