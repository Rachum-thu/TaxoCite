# Remote Sensing with Vision-Language Models

## Task Scoping and Formulations
**Description:** Problem framings that define inputs, outputs, and granularity for multimodal RS tasks.

### Image-Level Understanding Tasks
**Description:** Captioning, scene classification, and VQA with closed or open answer spaces.

### Region-Level Interaction Tasks
**Description:** Visual grounding, region captioning, and spatial reasoning over localized areas.

## Dataset Construction and Annotation
**Description:** Strategies to assemble multimodal corpora and supervision signals tailored for RS.

### Rule-Driven QA and Grounding Labels
**Description:** Programmatic question-answer and geometry generation from map data or annotations.

### LLM-Assisted Instruction Generation
**Description:** Few-shot prompting to create multi-turn task instructions and conversational answers.

## Pretraining Backbones and Modal Encoders
**Description:** Choices of foundation encoders and pretraining schemes adapted to RS imagery.

### Contrastive Vision-Language Encoders
**Description:** Dual encoders aligned on web-scale pairs, optionally enhanced for remote sensing.

### RS-Specific Self-Supervised Pretraining
**Description:** MAE-style transformers leveraging temporal cues and spectral bands without labels.

## Cross-Modal Fusion, Alignment, and Outputs
**Description:** Architectural components that align modalities and express predictions across formats.

### Attention-Based Fusion Blocks
**Description:** Self- and co-attention layers to model intra- and inter-modal dependencies.

### Connector and Prompt Mechanisms
**Description:** MLP adapters and learnable prompts to project visuals into language space.

### Textualization of Spatial Outputs
**Description:** Encoding boxes and masks as token sequences for language-only decoders.

## Adaptation and Tuning Regimes
**Description:** Methods to specialize pretrained models for RS tasks with limited labels.

### Prompt and Adapter Tuning
**Description:** Learnable prompts or lightweight modules for few-shot transfer and robustness.

### Instruction Tuning and Reinforcement Alignment
**Description:** Supervised dialogue fine-tuning and reward-guided consistency of reasoning and answers.

## Training Signals and Objectives
**Description:** Loss formulations and supervision designs that drive multimodal learning in RS.

### Contrastive Alignment and Cross-Entropy
**Description:** Image–text matching and categorical objectives for grounding and classification.

### Hybrid Grounding Supervision
**Description:** Converting masks/boxes to text plus consistency constraints across output types.

## Evaluation, Robustness, and Interpretability
**Description:** Practices to assess performance, explain behavior, and test generalization in RS.

### Task-Specific Metrics and Protocols
**Description:** Acc@0.5, BLEU/CIDEr, OA/AA, mIoU, and standardized split or threshold choices.

### Generalization and Data-Shift Assessment
**Description:** Zero/few-shot tests, cross-location splits, and temporal or sensor domain shifts.

### Explanations and Consistency Checks
**Description:** Attention rollout, agreement scores, refusal to answer, and logic-coherence measures.

## External Knowledge and Context Integration
**Description:** Mechanisms to augment visual input with retrieved or structured geospatial knowledge.

### Retrieval-Augmented Reasoning
**Description:** Vector databases and fused similarity to inject context into generation.

## Multispectral and Temporal Modeling
**Description:** Designs that exploit non-RGB bands and time series in vision-language pipelines.

### Spectral/Temporal Encoding and Masking
**Description:** Positional encodings, band grouping, and independent masking across time or spectra.

## Other Topics
**Description:** Topics and methods that do not fit into the above categories or are emerging areas.