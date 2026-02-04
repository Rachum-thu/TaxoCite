# generative retrieval

## Generative document retrieval
**Description:** Retrieves documents by generating identifiers using parametric memory and constrained decoding.

### Training strategies and data augmentation
**Description:** Improves indexing by supervised mapping and pseudo queries synthesized from documents.

### Architectural and decoding innovations
**Description:** Adapts model structures and decoding to better memorize corpora and generate valid IDs.

### Document identifiers: numeric-based
**Description:** Represents items with atomic, structured, or quantized numeric codes as targets.

### Document identifiers: text-based
**Description:** Uses titles, URLs, substrings, or term sets to uniquely denote items.

### Document identifiers: learnable representations
**Description:** Learns semantic identifiers jointly with models via reconstruction or reparameterization.

### Incremental learning on dynamic corpora
**Description:** Updates indices for new items while mitigating forgetting and preserving past performance.

### Downstream task adaptation
**Description:** Applies the paradigm to tasks like fact verification, QA, code, and dialogue.

### Multi-modal generative retrieval
**Description:** Extends identifier generation across modalities, aligning visual and textual features.

## Reliable response generation
**Description:** Produces user-centric answers directly, with strategies to enhance faithfulness and utility.

### Model structure enhancements
**Description:** Scales parameters and integrates experts or models to strengthen knowledge retention.

### Training and inference optimization
**Description:** Improves reliability via curated data, objectives, and refined inference procedures.

### Prompting and reasoning strategies
**Description:** Guides stepwise thinking and self-verification to boost correctness and transparency.

### Knowledge updating and editing
**Description:** Adds or modifies internal facts through continual learning and targeted parameter changes.

### Retrieval augmentation
**Description:** Grounds outputs with fetched evidence using sequential, branching, conditional, or iterative flows.

### Tool augmentation
**Description:** Invokes external services like search, APIs, graphs, or models to extend capabilities.

### Response generation with citations
**Description:** Produces attributions either intrinsically or supported by retrieved evidence sources.

### Personalized information assistants
**Description:** Tailors interactions to user profiles across dialogue and domain-specific applications.

## Evaluation
**Description:** Assesses systems via retrieval and generation metrics, benchmarks, and diagnostic analyses.

### Evaluation of generative document retrieval
**Description:** Uses ranking metrics and datasets to measure indexing quality and scalability.

### Evaluation of response generation
**Description:** Combines automatic, human, and factuality metrics on broad capability benchmarks.

## Challenges and unified frameworks
**Description:** Addresses open issues and integration directions for building end-to-end GenIR systems.

### Scalability and efficiency of generative retrieval
**Description:** Faces latency and accuracy limits on million-scale corpora and long identifier decoding.

### Dynamic corpora and incremental updates
**Description:** Requires robust strategies for continual indexing without catastrophic forgetting.

### Identifier design and decoding constraints
**Description:** Demands concise, unique, interpretable targets and efficient constrained generation.

### Factuality and timeliness in response generation
**Description:** Needs better grounding, real-time knowledge access, and trustworthy outputs.

### Unified retrieval-generation frameworks
**Description:** Aims to integrate docID generation, retrieval decisions, and answer synthesis.

## Other Topics
**Description:** Topics and methods that do not fit into the above categories or are emerging areas.