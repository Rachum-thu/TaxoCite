# point_cloud

## Input Representations and Modalities
**Description:** Summarizes how 3D data are formatted for learning and inference.

### Mesh-based representations
**Description:** Uses polygonal surfaces to encode geometry directly from connected vertices and faces.

### Projection-based representations
**Description:** Projects 3D samples to 2D views enabling image-style feature extraction and fusion.

### Volumetric voxelization
**Description:** Discretizes space into 3D grids to apply regular convolutional processing.

### Raw point-based representations
**Description:** Operates on unordered point sets while preserving permutation invariance and locality.

### Hybrid multi-modal representations
**Description:** Combines points, voxels, and views to leverage complementary spatial and contextual cues.

## 3D Shape Classification: Learning Strategies
**Description:** Method families used to predict whole-object labels from point clouds.

### Pointwise MLP architectures
**Description:** Share MLP weights across points then aggregate globally via symmetric pooling.

### Convolution-based operators
**Description:** Define discrete or continuous kernels to aggregate local geometric neighborhoods.

### Graph-based neural models
**Description:** Build dynamic graphs over points to learn edge-aware, relation-driven features.

### Transformer-based attention models
**Description:** Use self-attention to capture long-range dependencies with permutation-invariant design.

### Hierarchical structures
**Description:** Learn multi-scale features via set abstraction, trees, or layered subsampling.

### Sequential RNN-based models
**Description:** Process ordered beams or multi-scale sequences to encode contextual dependencies.

## Semantic Segmentation: Approaches
**Description:** Techniques assigning semantic labels to individual points in scenes.

### Projection-based segmentation
**Description:** Segments range, multi-view, or bird’s-eye projections then reprojects to 3D.

### Discretization-based segmentation
**Description:** Employs dense or sparse voxel partitions to enable efficient 3D convolutions.

### Hybrid fusion methods
**Description:** Integrates range, point, and voxel branches with learned cross-modal fusion.

### Raw point supervised architectures
**Description:** Apply MLP, convolution, graph, or transformer backbones directly to points.

### Weakly and unsupervised segmentation
**Description:** Leverages limited labels or self-supervision via contrastive, prototype, or consistency losses.

## Point Cloud Registration
**Description:** Estimates rigid transformations to align multi-view point sets into a common frame.

### Mathematical optimization-based registration
**Description:** Solves alignment via robust estimators, convex relaxations, or distribution-based models.

### Feature-based registration
**Description:** Matches descriptors like PFH, FPFH, SHOT, or structural cues before alignment.

### ICP-based refinement
**Description:** Iteratively minimizes point-to-point, plane, or line distances with enhanced variants.

### Deep learning-based registration
**Description:** Learns descriptors and end-to-end correspondences with attention and differentiable matching.

## Self-supervised and Cross-modal Pretraining
**Description:** Unlabeled or auxiliary-modality objectives to improve 3D representations.

### Generative modeling pretraining
**Description:** Uses autoencoders, VAEs, GANs, or capsules to reconstruct or synthesize shapes.

### Masked autoencoding and tokenization
**Description:** Predicts masked patches or tokens to learn local-global geometric priors.

### Contrastive and cross-modal objectives
**Description:** Aligns views, time, language, or images with contrastive or teacher-student schemes.