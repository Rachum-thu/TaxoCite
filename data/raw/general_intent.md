# Citation Intent

## Background
**Description:** Citations used to frame the work: introduce context, define the problem, motivate why it matters, identify gaps, or summarize prior lines of work without directly adopting them as components or evaluation instruments.

### Domain Overview
**Description:** Provide broad background about the research area, common concepts, or a high-level landscape of the domain.

### Problem Formulation
**Description:** Define the task/problem, formal setting, assumptions, terminology, or notation; establish “what is being solved” rather than “how it is evaluated.”

### Importance of Problem
**Description:** Justify why the problem is meaningful or challenging, including societal/scientific value or practical urgency.

### Prospective Application
**Description:** Motivate the work by pointing to downstream applications, use-cases, or potential impact areas enabled by solving the problem.

### Research Gap
**Description:** Identify limitations, missing capabilities, or open challenges in prior work that motivate the proposed approach.

### Prior Methods
**Description:** Summarize or categorize existing approaches as prior art (e.g., common method families) without directly using them as a component, baseline, or protocol.

## Method
**Description:** Citations used as part of the proposed approach: adopting, adapting, or building on prior methods/resources as components in the system or pipeline.

### Model/Architecture Adoption
**Description:** Adopt a specific model family, architecture design, or modeling framework from prior work as a component (often with minimal modification).

### Algorithm/Principle Adoption
**Description:** Adopt an algorithmic procedure, objective, principle, or technique (e.g., training trick, inference strategy, optimization method) from prior work.

### Resource Utilization
**Description:** Use prior artifacts as inputs or infrastructure for the method (e.g., external tools, libraries, knowledge bases, datasets used as training data, or released code used to implement the method), excluding cases where the resource is used primarily for evaluation.

## Evaluation
**Description:** Citations used to justify, define, or instantiate how performance is measured and compared, including datasets/benchmarks used for testing, metrics, baselines, and experimental protocols.

### Result Comparison
**Description:** Use a cited system/method as a baseline comparator, or cite prior numbers as a reference point for comparison (including SOTA claims).

### Benchmark Utilization
**Description:** Use a cited dataset/benchmark/suite primarily as an evaluation testbed (data splits, tasks, benchmark rules), not as training data.

### Metrics Utilization
**Description:** Use a cited metric, scoring rule, judge, or evaluation criterion (including model-based evaluation/judging protocols) to measure correctness, quality, or other outcomes.

### Hyperparameter Utilization
**Description:** Adopt hyperparameter choices, training/evaluation settings, or implementation details from prior work primarily to reproduce or standardize experimental conditions.

### Setting/Protocal Adoption
**Description:** Adopt an experimental setup or protocol from prior work (e.g., evaluation pipeline, prompting/evaluation procedure, data preprocessing protocol, fairness/robustness protocol), focusing on *how evaluation is conducted* rather than the method itself.

## Other Intent
**Description:** Any citation role not covered above or cases where the local context is insufficient to confidently map to a node (e.g., acknowledgements/credits, dataset license notes, peripheral mentions, unclear function).
