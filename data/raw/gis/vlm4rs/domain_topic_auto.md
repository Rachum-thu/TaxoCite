# Remote Sensing with Vision-Language Models

## Task Formulations in RS VLM
**Description:** Canonical problem settings defining what models must perceive, reason, and output.

### Image-Level Understanding
**Description:** Scene description and categorization using captions or class names for entire images.

### Object/Region-Level Grounding and Referring
**Description:** Localizing entities from text queries with boxes, orientations, or regions.

### Pixel-Level and Dense Outputs
**Description:** Producing masks or per-pixel labels via text-conditioned segmentation or change maps.

### Temporal and Multimodal Variants
**Description:** Extending tasks to time series, multispectral bands, and cross-modality reasoning.

## Dataset Construction and Supervision Sources
**Description:** Strategies for assembling paired vision-language data and deriving supervisory signals.

### Rule-Based and OSM-Derived QA Pairs
**Description:** Automatic question-answer generation from geographic annotations, tags, and heuristics.

### Visual Grounding Corpora and Geometry Labels
**Description:** Collections with HBB, OBB, and masks enabling location- and shape-aware supervision.

### Instruction-Following Conversational Sets
**Description:** Multitask, multi-turn prompts and answers crafted or synthesized for LLM alignment.

### Train/Val/Test Splits and Domain-Shift Protocols
**Description:** Geographic, sensor, and temporal partitions to study generalization and robustness.

## Cross-Modal Alignment and Fusion
**Description:** Mechanisms aligning image and text representations to enable joint reasoning.

### Contrastive Alignment Objectives
**Description:** Learning shared spaces where paired image-text embeddings are maximally similar.

### Attention-Based Co-Encoding
**Description:** Self- and cross-attention modeling intra- and inter-modal dependencies for fusion.

### Vision-Language Connectors to LLMs
**Description:** Linear, MLP, or Q-Former adapters mapping visual tokens into language spaces.

### Prompt-Based Modulation
**Description:** Learnable or handcrafted prompts steering alignment and conditioning computation.

## Representation Learning Backbones for RS
**Description:** Pretraining choices and encoders tailored to remote sensing signal characteristics.

### Self-Supervised Pretraining on RS Data
**Description:** Masked reconstruction or contrastive schemes leveraging unlabeled satellite imagery.

### Multispectral and Temporal Encodings
**Description:** Positional schemes and masking across bands or time to preserve structure.

### Vision Encoder Choices and Scales
**Description:** ViT, EVA, XCiT, or CNN backbones adapted to small objects and large contexts.

### Language Models and Tokenization
**Description:** BERT-style encoders and LLMs (e.g., Vicuna, LLaMA) handling RS-specific text.

## Training and Adaptation Regimes
**Description:** Procedures for transferring, aligning, and specializing models to RS tasks.

### Instruction Tuning and SFT
**Description:** Supervised fine-tuning on instruction-following pairs to elicit task behavior.

### Parameter-Efficient Adaptation
**Description:** LoRA, adapters, and shared prompts minimizing trainable weights and memory.

### Hybrid Supervision and Consistency Losses
**Description:** Prompt-assisted, geometry-guided, and cross-signal constraints enforcing agreement.

### Knowledge Distillation and Freezing Policies
**Description:** Teacher-student transfer and selective freezing to retain general capabilities.

## Output Structuring and Interface Design
**Description:** Conventions for representing predictions and interacting with model inputs.

### Textualization of Geometry and Masks
**Description:** Encoding HBB, OBB, and downsampled masks as compact token sequences.

### Answer Space Design and Class Taxonomy
**Description:** Closed-set vocabularies, quantized counts, and numerical ranges for QA.

### Region and Point Prompting Interfaces
**Description:** Coordinate I/O formats and visual prompts directing localized reasoning.

## Evaluation Protocols and Diagnostics
**Description:** Quantitative metrics, analyses, and tools for validating capabilities and limits.

### Metrics and Thresholds for Grounding
**Description:** IoU@0.5, rotated IoU, and consistency scores assessing localization fidelity.

### Captioning and QA Scoring
**Description:** BLEU, ROUGE, METEOR, CIDEr, OA/AA quantifying linguistic and classification quality.

### Ablation and Component Attribution
**Description:** Controlled removals and sensitivity analyses isolating contributions of modules.

### Explainability and Attention Visualization
**Description:** Attention rollout and maps to interpret cross-modal focus and evidence.

## Efficiency and Resource Considerations
**Description:** Design trade-offs enabling practical training and deployment at scale.

### Lightweight Encoder Substitutions
**Description:** Tiny or mobile transformers replacing heavy backbones with minimal accuracy loss.

### Memory/Compute Optimization Techniques
**Description:** LoRA ranks, ZeRO stages, and batch schemes reducing hardware footprint.

### Multiresolution and Patch Strategies
**Description:** Token downsampling, resized inputs, and positional interpolation for throughput.

## Reliability, Reasoning, and Trustworthiness
**Description:** Methods ensuring models provide correct answers for the right reasons.

### Logical Consistency Reinforcement
**Description:** Rewards aligning chain-of-thought with outcomes under option permutations.

### Hallucination Mitigation and Honesty
**Description:** Refusals, self-awareness, and deception-aware data to curb fabricated content.

### Bias Analysis and Error Taxonomies
**Description:** Probing language priors, counting failures, and positional confusions in RS VQA.

## Knowledge Augmentation and External Context
**Description:** Integrating retrieved facts and world knowledge to ground responses.

### Multimodal Knowledge Bases and Retrieval
**Description:** Vector databases indexing images and texts for cross-modal nearest-neighbor search.

### Knowledge-Conditioned Prompt Construction
**Description:** Retrieved snippets fused into prompts guiding context-aware generation.

### Evaluation on Knowledge-Intensive Tasks
**Description:** Benchmarks requiring historical, cultural, or environmental background reasoning.

## Robustness, Domain Shift, and Generalization
**Description:** Coping with changes across geography, sensors, and distributions.

### Cross-Sensor and Cross-Region Transfer
**Description:** Testing on unseen locations or platforms to assess adaptability.

### Zero-/Few-Shot and Prompt Sensitivity
**Description:** Performance under minimal labels and template variations for open-set classes.

### Small Object Density and Counting Challenges
**Description:** Handling scale variance, crowded scenes, and quantized count regimes.

## Other Topics
**Description:** Topics that do not fit into the above categories or are emerging areas.