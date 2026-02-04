
# Seed Data Curation — Topic + Papers + Surveys

This section specifies the manual “seed collection” stage prior to any automated dataset construction. The output of this stage is a topic folder containing PDFs of surveys and papers.

## 0) Read the paper
Before collecting anything, the annotator should read (or at least, understand) the **Introduction** of the task paper to know the task setting.

## 1) Choose a topic (survey-grounded)
Each annotator selects a topic that:
- Belongs to one of the allowed domains: **cs**, **gis**, **chemistry**
- Has **survey coverage** (surveys define the scope of the topic)
- Find **2–3 survey papers** that cover the topic and download their **PDFs**.
Rationale:
- The surveys will be used to induce the domain topic taxonomy later.

## 2) Select ~10 non-survey papers (moderate citation count)
For the same topic, select approximately **10 representative non-survey papers**, with a **moderate** citation count to keep later annotation manageable:
- Suggested citation range: **~20–60 citations**
- Extremely highly cited papers tend to be annotation-heavy (too many references/candidates)

## 3) Create the directory structure
Create a new folder at:
- `data/raw/{domain}/{topic}/`

Example:
- `data/raw/cs/rag/`

Within that folder, store PDFs in dedicated subfolders:
- `data/raw/{domain}/{topic}/pdf_papers/`   (the ~10 non-survey papers)
- `data/raw/{domain}/{topic}/pdf_surveys/`  (the 2–3 surveys)

Example:
- `data/raw/cs/retrieval_augmented_generation/pdf_papers/`
- `data/raw/cs/retrieval_augmented_generation/pdf_surveys/`

No additional metadata file is required (PDFs only).

# Automatic Construction + Manual Review (Internal)

This section describes the fully automated construction pipeline (triggered by one shell script) and the required manual spot-checks to verify the outputs are usable.

## 0) One-command entry point
After the manual seed PDFs are prepared under `data/raw/{domain}/{topic}/`, run:

- `bash data/raw/auto_construct.sh <directory_path>`

You should run the one below if your paper is not using ACM-like reference markers.

- `bash data/raw/auto_construct_no_meta.sh <directory_path>`

Example:
- `bash data/raw/auto_construct.sh data/raw/cs/retrieval_augmented_generation`

You do **not** need to run the individual Python utilities manually; the script orchestrates all steps.

**IMPORTANT**:
- All valid/expected outputs and naming convention for each step should be found as examples under:
  - `data/raw/cs/retrieval_augmented_generation/`

## 1) Step 1 — PDF → TXT (OCR / extraction)
The pipeline first converts PDFs into plain text.

Code:
- `data/raw/util/pdf2txt.py`

Outputs:
- `data/raw/{domain}/{topic}/txt_papers/`
- `data/raw/{domain}/{topic}/txt_surveys/`

Expected issues:
- Typically stable; usually does not require intervention.

Manual check (spot-check):
- Open a few `.txt` files from both `txt_papers/` and `txt_surveys/`.
- Confirm the text is not empty, not truncated, and includes the References section for papers.

## 2) Step 2 — TXT → Markdown normalization
Because OCR/plain-text formatting is often messy, the pipeline calls an LLM to rewrite the text into a cleaner Markdown structure.

Code:
- `data/raw/util/txt2md.py`

Outputs:
- `data/raw/{domain}/{topic}/md_papers/`
- `data/raw/{domain}/{topic}/md_surveys/`

Manual check (spot-check):
- Verify headings, paragraphs, lists, and section structure are reasonably preserved.
- Confirm that the References section (if present) is still present and readable.

## 3) Step 3 — Extract references from each paper
The pipeline extracts structured reference entries from each paper’s appended References list, using an LLM.

What is extracted (per reference entry):
- A **unique citation marker** as used in the paper (e.g., numeric markers like `[14]` in ACM/ACM-like formats)
- The **referenced paper title**

Code:
- `data/raw/util/txt2ref.py`

Outputs:
- `data/raw/{domain}/{topic}/yaml_refs/`

Critical requirement:
- The paper’s reference list must contain a citation marker that corresponds to the in-text marker (i.e., the “unique marker” is explicitly visible in the reference list).
- If the marker is not explicitly present (or mapping requires hyperlink navigation), extraction may fail or be unreliable. In that case, you may need to adjust the prompt and/or use a stronger model.

Manual check (spot-check):
- Pick 1–2 papers and open their YAML reference outputs.
- Confirm: (a) markers are present, (b) titles look correct, (c) the number of extracted references is plausible (not ~0, not wildly off).

## 4) Step 4 — Induce topic taxonomy from surveys
Using the survey Markdown files, the pipeline induces a domain/topic taxonomy.

Code:
- `data/raw/util/surveys2taxo.py`

Outputs:
- `data/raw/{domain}/{topic}/domain_topic.md`

Manual check (spot-check):
- Verify the taxonomy is relevant to the chosen topic and not generic/noisy.
- Confirm it resembles a hierarchy suitable for later label-space induction (coarse → fine).

## 5) Step 5 — Segment papers (Markdown → segments)
To avoid showing a full paper during later classification/annotation, the pipeline segments each paper’s Markdown into smaller units.

Code:
- `data/raw/util/md2seg.py`

Outputs:
- `data/raw/{domain}/{topic}/seg_papers/`

Manual check (spot-check):
- Open a few segments from a few papers.
- Confirm segments are coherent (not mid-word, not totally broken) and roughly align with logical local contexts.

## 6) Step 6 — Locate citations / attach citation-local context
The pipeline then locates references/citations in the segmented content (LLM- or rule-assisted) and appends/updates reference metadata accordingly.

Code:
- `data/raw/util/locate_ref.py`

Outputs:
- `data/raw/{domain}/{topic}/yaml_refs/` (appended/updated relative to Step 3)

Manual check (spot-check):
- For a few citations, verify the located markers match the expected references.
- Ensure citation-local contexts are being attached where expected (and not systematically missing).

# LM Pre-Annotation + Human Correction (Internal)

This stage produces draft labels using an LLM, then requires a human annotator to verify and correct them.

## 0) One-command entry point
Run the auto-annotation script on the constructed dataset directory:

- `bash data/raw/auto_annotate.sh $DIR`

Example:
- `bash data/raw/auto_annotate.sh data/raw/cs/retrieval_augmented_generation`

## 1) What the script does (high level)
The script performs LLM-based classification over the constructed data (segmented contexts + located citations). It generates a “human review” workload where each citation mention in a text block is assigned draft labels.

## 2) Output: human_todo folder (what humans edit)
The script creates a folder:

- `data/raw/{domain}/{topic}/human_todo/`

Inside, you will see YAML files per paper, e.g.:
- `data/raw/cs/retrieval_augmented_generation/human_todo/paper_1.yaml`

Each YAML contains multiple text blocks, each with:
- `block_id`: unique block index
- `content`: the citation-local context text shown to the annotator
- `citations`: a list of citation markers found in this block, each with draft labels

Example structure (illustrative):

- `block_id: 1`
- `content: <text block with citation markers like [5], [9], ...>`
- `citations:`
  - `marker: "[5]"`
    - `intent_label: <draft intent label>`
    - `topic_label: <draft topic label>`
  - `marker: "[9]"`
    - `intent_label: <draft intent label>`
    - `topic_label: <draft topic label>`
  - ...

## 3) Human correction procedure (what you need to do)
Before editing, read the two taxonomies:
- The **intent taxonomy** (the intent label space)
- The **topic taxonomy** (the domain/topic label space induced from surveys)

Then, create a folder named `human_annotation/`.

Then, for each YAML file under `human_todo/`:

0. copy it to the human_annotation folder.
1. Read the `content` block carefully.
2. For each `marker` under `citations`, verify whether the LLM’s:
   - `intent_label` and
   - `topic_label`
   are correct **given the context and the taxonomy definitions**.
3. If correct, leave it unchanged.
4. If incorrect or suboptimal, replace with the best label(s) from the taxonomies.
