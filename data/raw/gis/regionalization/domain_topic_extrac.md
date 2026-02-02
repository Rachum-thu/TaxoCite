# regionalization

## Problem formulations and criteria
**Description:** Defines task variants, objective functions, and constraint families guiding region design**

### p-regions formulation
**Description:** Partitions units into a fixed number of contiguous regions minimizing within-region heterogeneity.**

### max-p regions formulation
**Description:** Determines the maximum feasible number of regions satisfying minimum size and similar constraints.**

### Heterogeneity objectives
**Description:** Measures internal dissimilarity via pairwise differences or deviations from regional means.**

### Enriched aggregate constraints
**Description:** Supports user-defined per-region thresholds and multiple simultaneous aggregate conditions.**

## Indirect contiguity approaches
**Description:** Satisfy connectivity implicitly through compactness or post-processing rather than explicit constraints.**

### Conventional clustering with a posteriori contiguity correction
**Description:** Runs nonspatial clustering, then splits disconnected clusters into contiguous regions.**

### Compactness-driven location–allocation models
**Description:** Assigns areas to fixed centers minimizing weighted distances with balance requirements.**

### Multiobjective clustering including spatial coordinates
**Description:** Combines attribute similarity with geographic proximity to induce compact, contiguous groupings.**

## Exact optimization models
**Description:** Encode objectives and connectivity constraints in mathematical programs solved exactly.**

### Adjacency-level assignment constraints
**Description:** Uses network-based adjacency levels from centers to enforce connectivity during assignment.**

### MTZ ordering constraints for contiguity
**Description:** Adapts traveling-salesman ordering variables to prevent cycles and maintain connectivity.**

### Core-order assignment strategies
**Description:** Assigns areas by increasing contiguity order from designated regional cores.**

### Network-flow connectivity enforcement
**Description:** Models regions as connected flows from multiple sources to a single sink.**

### Enumerative set-partitioning with column generation
**Description:** Selects feasible regions from a generated pool using linear optimization techniques.**

## Heuristic models with explicit contiguity
**Description:** Construct or refine partitions using search procedures while maintaining connectivity.**

### Contiguity-constrained hierarchical agglomeration
**Description:** Merges only adjacent clusters within modified linkage algorithms to build nested solutions.**

### Seed-and-grow region construction
**Description:** Initiates regions at seeds and accretes neighboring areas under stopping rules.**

### Local improvement via area swapping
**Description:** Iteratively reassigns boundary units, allowing simulated annealing or tabu moves.**

### Top-down divisive MST methods
**Description:** Cuts high-cost edges in spanning trees to split space into homogeneous partitions.**

### Spanning-tree partitioning heuristics
**Description:** Replaces deleted links within spanning trees to iteratively improve connected partitions.**

## Hybrid heuristic–exact methods
**Description:** Combine localized exact optimization with heuristics to balance quality and scalability.**

### Selective local exact refinement
**Description:** Optimizes subsets of neighboring regions with exact solvers to enhance global quality.**

### Heuristic concentration for model reduction
**Description:** Uses multiple heuristic solutions to drastically shrink exact model size.**

### Distillation for flow-based formulations
**Description:** Reduces flow models by retaining variables supported across best heuristic partitions.**

## Learning-based methods
**Description:** Leverage representation learning on adjacency graphs to derive contiguous clusters.**

### Graph-embedding with contiguity-aware clustering
**Description:** Learns unsupervised embeddings, then clusters while restricting merges to adjacent units.**

### Kernel-and-extension heuristics
**Description:** Seeds with cohesive communities and greedily grow regions under size and compactness.**

### Joint-loss representation learning
**Description:** Trains models to reward attribute similarity and interaction intensity among neighbors.**