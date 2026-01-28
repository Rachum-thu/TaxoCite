# retrieval augmented generation

## Retrieval
**Description:** Design choices for accessing external knowledge, balancing relevance, efficiency, and adaptability.

### Retriever Type
**Description:** Sparse keyword matching versus trainable dense encoders for semantic similarity retrieval.

### Retrieval Granularity
**Description:** Selection of indexing units such as documents, chunks, tokens, entities, or mentions.

### Indexing and Query Optimization
**Description:** Chunking, metadata, hierarchical indices, and query rewriting/expansion to improve recall and precision.

## Generation
**Description:** Language modeling approaches that utilize retrieved context to produce accurate, coherent outputs.

### Parameter-Accessible Generators (White-box)
**Description:** Encoder–decoder or decoder-only models fine-tuned to leverage external context.

### Parameter-Inaccessible Generators (Black-box)
**Description:** Closed models augmented via prompts and examples without parameter updates.

### Context Curation and Compression
**Description:** Reranking, selection, and summarization to reduce noise and mitigate long-context limitations.

## Retrieval Integration for Generation
**Description:** Mechanisms connecting retrieval outputs with generators at different processing stages.

### Input-Layer Integration
**Description:** Concatenating or parallel-encoding retrieved texts with queries as model inputs.

### Output-Layer Integration
**Description:** Post-hoc fusion of next-token distributions with retrieval-induced neighbor signals.

### Intermediate-Layer Integration
**Description:** Cross-attention or memory modules injecting retrieved representations into hidden states.

## Training Strategies
**Description:** Paradigms for optimizing retrievers and generators to effectively use external knowledge.

### Training-free Methods
**Description:** Inference-time augmentation via prompt engineering or retrieval-guided token calibration.

### Independent Training
**Description:** Separate optimization of retriever and generator without mutual interaction.

### Sequential Training
**Description:** Pretrain one module, then tune the other under fixed counterpart supervision.

### Joint Training
**Description:** End-to-end learning aligning retrieval and generation, often with indexed corpora.

## Applications
**Description:** Practical use cases benefiting from augmented knowledge access and grounded generation.

### NLP Applications
**Description:** Open-domain QA, conversational agents, and fact verification enhanced by retrieved evidence.

### Downstream Tasks
**Description:** Broader tasks like recommendation, code generation, and text-to-SQL with retrieval support.

### Domain-specific Applications
**Description:** Specialized areas such as molecular science and finance leveraging curated external corpora.

## Evaluation
**Description:** Assessment of retrieval relevance and generation fidelity across tasks and benchmarks.

### Retrieval Quality
**Description:** IR-style metrics evaluating relevance, ranking, and coverage of retrieved context.

### Generation Quality
**Description:** Measures of factuality, relevance, safety, and task accuracy for produced answers.

### Evaluation Aspects and Tools
**Description:** Quality scores and robustness abilities assessed via benchmarks and automated evaluators.

## Challenges and Future Directions
**Description:** Open issues around reliability, scaling, and deployment of augmented language systems.

### Trustworthiness and Robustness
**Description:** Ensuring robustness, fairness, explainability, and privacy against manipulation and leakage.

### Scaling and Modality Expansion
**Description:** Adapting to multilingual, multimodal, long-context, and production-ready hybrid RAG–FT workflows.