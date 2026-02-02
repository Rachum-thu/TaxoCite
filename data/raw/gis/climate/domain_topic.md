# climate

## Data types and structures
**Description:** Core modalities capturing atmospheric information for analysis and modeling.

### Time series representations
**Description:** Sequential observations at individual sites with one or multiple meteorological variables.

### Gridded spatio-temporal fields
**Description:** Multi-level global or regional arrays on latitude–longitude grids evolving over time.

### Climate text corpora
**Description:** Documents and reports enabling topic, sentiment, and policy analyses in climate discourse.

## Modeling approaches and paradigms
**Description:** High-level strategies framing how future states are predicted from data.

### Direct forecasting
**Description:** One-step prediction for fixed horizons using only current conditions as input.

### Continuous lead-time conditioning
**Description:** Lead time provided as an input to generate forecasts at arbitrary horizons.

### Iterative dynamics forecasting
**Description:** Short-interval rollouts predicting state changes to extend to long horizons.

## Architecture families for climate modeling
**Description:** Principal neural model classes tailored to spatio-temporal and textual climate data.

### Recurrent networks and ConvLSTM
**Description:** Sequence models capturing temporal dependencies, often with convolution for spatial structure.

### Transformers for climate modeling
**Description:** Attention-based architectures modeling long-range spatio-temporal dependencies across variables.

### Graph neural networks for Earth systems
**Description:** Message-passing over learned or geodesic meshes to encode spatial relations and dynamics.

### Generative models: GANs and diffusion
**Description:** Probabilistic generators for sharp sequences, ensembles, or residual refinement of fields.

## Learning and training strategies
**Description:** Techniques for efficient pretraining, adaptation, and constraint-aware optimization.

### Supervised pretraining and finetuning of foundation models
**Description:** Large models trained on reanalysis, then adapted to diverse downstream tasks.

### Self- and semi-supervised learning for climate
**Description:** Masked reconstruction and pseudo-labeling leverage unlabeled data for representations.

### Federated learning and privacy-preserving training
**Description:** Collaborative optimization across institutions without exchanging sensitive raw data.

### Physics-informed objectives and constraints
**Description:** Loss designs embedding latitude weighting, pressure weighting, or governing equations.

## Forecasting and simulation tasks
**Description:** Core predictive targets across temporal ranges and spatial scales.

### Short- to medium-range global prediction
**Description:** Multi-day forecasts of atmospheric variables on global or regional grids.

### Precipitation nowcasting
**Description:** Minute-to-hour prediction of rainfall fields from radar and satellite sequences.

### Subseasonal-to-seasonal and climate projection
**Description:** Weeks-to-months forecasting and long-term trend simulation for planning and risk.

## Post-processing and enhancement tasks
**Description:** Methods improving resolution, realism, and interpretability of model outputs.

### Downscaling and super-resolution
**Description:** Increasing spatial detail of coarse outputs while preserving physical consistency.

### Bias correction and calibration
**Description:** Adjusting systematic errors to align predictions with observational distributions.

### Weather pattern detection and understanding
**Description:** Identifying phenomena like cyclones, rivers, or oscillations for process insight.

## Uncertainty and inference strategies
**Description:** Approaches to quantify, reduce, and manage prediction uncertainty.

### Multi-interval ensembling via randomized rollouts
**Description:** Averaging diverse interval combinations to stabilize long-horizon forecasts.

### Probabilistic forecasting with diffusion or variational objectives
**Description:** Learning distributions to sample multiple plausible future scenarios.

### Multi-step training to mitigate error accumulation
**Description:** Finetuning with rolled-out objectives to reduce compounding prediction errors.

## Challenges and future directions
**Description:** Outstanding issues guiding research toward reliable, general, and responsible models.

### Multimodal integration across heterogeneous resolutions
**Description:** Fusing radar, satellite, text, and reanalysis with differing scales and cadences.

### Interpretability and causal reasoning in learned models
**Description:** Explaining decisions and aligning attributions with physical mechanisms.

### Robust generalization under non-stationary climate
**Description:** Maintaining skill under shifting baselines and extreme event distributions.

### Privacy, communication costs, and on-device adaptation
**Description:** Balancing data protection, bandwidth limits, and local continual learning.