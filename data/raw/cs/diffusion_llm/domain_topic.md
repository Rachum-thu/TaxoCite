# diffusion llm

## Pre-trained language models
**Description:** Surveyed pretrained transformer families enabling contextual representations across NLP tasks.

### Encoder-only transformers
**Description:** Bidirectional encoders producing contextual token embeddings for understanding-focused applications.

### Decoder-only transformers
**Description:** Autoregressive generators emphasizing next-token prediction for language understanding via prompting.

### Encoder–decoder PLMs
**Description:** Sequence-to-sequence architectures pre-trained by denoising and text-to-text objectives.

### Long-context and memory extensions
**Description:** Mechanisms for handling extended sequences, including recurrence and relative positions.

## Deep learning foundations for text classification
**Description:** Neural architectures preceding PLMs that extract features directly from raw text.

### Convolutional approaches
**Description:** CNN-based models capturing local n-gram patterns with pooling operations.

### Recurrent and hybrid networks
**Description:** RNN, BiLSTM, and RCNN architectures modeling sequential dependencies and context.

### Capsule-based architectures
**Description:** Routing mechanisms capturing part–whole relationships for sentence representations.

### Graph neural methods for text
**Description:** Document and word graph modeling to learn relational textual structures.

## PLM-based fine-tuning strategies
**Description:** Approaches adapting general-purpose pretrained representations to downstream classification.

### Feature extraction with linear classifiers
**Description:** Using pooled representations with shallow heads to assign category labels.

### Contrastive representation learning
**Description:** Objectives pulling semantically similar texts together in embedding spaces.

### Semi-supervised and adversarial adaptation
**Description:** Fine-tuning with unlabeled data and GAN-style training to improve robustness.

### Label-aware and prototype methods
**Description:** Incorporating label semantics, prototypes, and expanded label spaces for supervision.

## Prompt-driven classification paradigms
**Description:** Techniques reformulating labeling as masked or generative prompts with minimal supervision.

### Zero-shot and few-shot prompting
**Description:** Template-based inference leveraging pretraining without extensive task-specific updates.

### Prompt tuning and meta-learning
**Description:** Lightweight parameterization and episodic training to generalize across tasks.

### Consistency and contrastive regularization
**Description:** Stabilizing predictions via explicit agreement and distance-based objectives.

### Retrieval- and knowledge-augmented prompting
**Description:** Enhancing prompts with external corpora, keywords, or knowledge graphs.

## Datasets and evaluation methodology
**Description:** Resources and metrics commonly used to train and assess classifiers.

### Sentiment and opinion benchmarks
**Description:** Corpora of labeled reviews and polarity judgments for sentiment tasks.

### Topic and news categorization corpora
**Description:** Datasets organizing articles into themes for multi-class classification.

### Question and short-text datasets
**Description:** Collections targeting intent, subjectivity, or question-type labeling.

### Accuracy, precision–recall, and F1 assessment
**Description:** Standard metrics summarizing classification correctness and balance.

## Other Topics
**Description:** Topics and methods that do not fit into the above categories or are emerging areas.