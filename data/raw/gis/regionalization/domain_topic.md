# regionalization

## Problem formulations and objectives
**Description:** Formulations specifying region count, constraints, and heterogeneity objectives guiding partition quality.

### p-regions formulations
**Description:** Fixed number of regions specified, optimizing intra-region homogeneity under constraints.

### max-p regions formulations
**Description:** Number of regions determined by optimization, maximizing feasible regions satisfying minimum requirements.

### Heterogeneity objective functions
**Description:** Homogeneity quantified via pairwise dissimilarities or deviations from regional means.

### Enriched aggregate constraints
**Description:** User-defined thresholds enforce size, balance, or aggregates per region during optimization.

## Algorithms without explicit spatial contiguity constraint
**Description:** Approaches indirectly enforce contiguity, often corrected a posteriori after aggregation.

### Conventional clustering with post hoc contiguity checks
**Description:** Two-stage clustering then split disconnected clusters into contiguous regions as needed.

### Compactness-driven location–allocation approaches
**Description:** Assign areas to predefined centers minimizing inertia, emphasizing compactness and equality.

### Multiobjective compactness–homogeneity clustering
**Description:** Combine geographic coordinates with attributes to balance shape compactness and internal similarity.

## Exact optimization models
**Description:** Mathematical programming formulations explicitly encode adjacency to guarantee region contiguity.

### Adjacency-level assignment formulations
**Description:** Control contiguity by constraining assignments through hierarchical neighbor levels from centers.

### Link-selection with subtour elimination
**Description:** Select inter-area links ensuring single connected components using subtour-breaking constraints.

### MTZ-order and numbering-based formulations
**Description:** Impose connectivity by ordering areas with lifted MTZ constraints adapted from routing.

### Order-based core assignment models
**Description:** Allocate areas by increasing contiguity order from designated core areas.

### Flow-based network connectivity formulations
**Description:** Model contiguity as feasible multi-source flows to single sinks within regions.

## Heuristic models with explicit contiguity
**Description:** Approximate algorithms preserving connectivity while constructing or improving spatial partitions.

### Top-down divisive edge-removal methods
**Description:** Iteratively cut high-cost edges on trees to produce increasingly homogeneous components.

### Bottom-up seeded agglomeration
**Description:** Grow regions from initial seeds by adding adjacent units satisfying constraints.

### Local search with area swaps and metaheuristics
**Description:** Iteratively reassign boundary units, allowing nonimproving moves via tabu or annealing.

### Contiguity-constrained hierarchical clustering
**Description:** Modify agglomerative linkage to merge only touching clusters across successive scales.

## Mixed heuristic–exact models
**Description:** Hybrid frameworks couple heuristics with exact solvers for localized, tractable improvements.

### Selective local reoptimization of neighboring regions
**Description:** Periodically re-solve subproblems for adjacent regions to refine shared boundaries.

### Heuristic concentration and model reduction
**Description:** Leverage multiple heuristic solutions to fix structure, shrinking exact model size.

### Distillation-based reduction for flow formulations
**Description:** Derive constraints from consensus assignments to simplify flow-based exact models.

## Learning-based methods
**Description:** Representation-learning pipelines discover contiguous groups using graph-structured attributes.

### Contiguity-aware graph embeddings with attribute and interaction signals
**Description:** Learn low-dimensional polygon representations combining features and flows on adjacency graphs.

### Kernel-extension growth from learned communities
**Description:** Seed regions with cohesive kernels, then greedily expand under size and compactness.

### Clustering in latent space with contiguity constraints
**Description:** Partition embeddings while enforcing adjacency, producing connected regions after training.