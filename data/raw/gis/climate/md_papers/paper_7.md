# FourCastNet 3: A geometric approach to probabilistic
machine-learning weather forecasting at scale

## Abstract
FourCastNet 3 advances global weather modeling by implementing a scalable, geometric machine
learning (ML) approach to probabilistic ensemble forecasting. The approach is designed to respect
spherical geometry and to accurately model the spatially correlated probabilistic nature of the
problem, resulting in stable spectra and realistic dynamics across multiple scales. FourCastNet 3
delivers forecasting accuracy that surpasses leading conventional ensemble models and rivals the best
di!usion-based methods, while producing forecasts 8 to 60 times faster than these approaches. In
contrast to other ML approaches, FourCastNet 3 demonstrates excellent probabilistic calibration
and retains realistic spectra, even at extended lead times of up to 60 days. All of these advances
are realized using a purely convolutional neural network architecture tailored for spherical geometry.
Scalable and e”cient large-scale training on 1024 GPUs and more is enabled by a novel training
paradigm for combined model- and data-parallelism, inspired by domain decomposition methods in
classical numerical models. Additionally, FourCastNet 3 enables rapid inference on a single GPU,
producing a 60-day global forecast at 0.25 °, 6-hourly resolution in under 4 minutes. Its computational
e”ciency, medium-range probabilistic skill, spectral ﬁdelity, and rollout stability at subseasonal
timescales make it a strong candidate for improving meteorological forecasting and early warning
systems through large ensemble predictions.

## 1 Introduction
Numerical weather prediction (NWP) is central to modern meteorology, underpinning our ability to
accurately understand and forecast atmospheric phenomena [ 1]. Advances in mathematical modeling,
computational power, and data assimilation have made NWP essential for weather forecasting, hazard
mitigation, energy management, and climate studies.
Traditional NWP models, however, are computationally intensive, limiting their ability to deliver rapid,
large-scale probabilistic forecasts. Recently, machine learning (ML) approaches have surpassed traditional
NWP in forecast skill and speed, enabling rapid generation of large ensembles and opening new possibilities
for weather and climate prediction [ 2–6]. These advances support improved sampling of rare events and
longer-range forecasts [ 7–9]. Despite these beneﬁts, ML models face challenges: they may struggle with
out-of-distribution events, physical consistency, and long-term stability [ 10–14]. Furthermore, commonly
used evaluation metrics fail to fully capture the accuracy with which these models approximate the
underlying dynamical systems [ 15]. Hybrid models that combine ML and traditional NWP o!er partial
solutions [ 16], but su!er from the same computational bottlenecks due to the Courant-Friedrichs-Lewy
(CFL) condition, making them expensive to evaluate, especially at high resolution. Additionally,
deterministic ML models often exhibit excessive smoothing, which are closer to ensemble averages,
lacking the ﬁdelity of traditional deterministic forecasts [ 17–19].
Recently, probabilistic approaches have aimed to address the latter problem [ 6, 20–22]. GenCast [ 6],
the state-of-the-art probabilistic ML model, has proven that a denoising di!usion model approach [ 23]
is e!ective at modeling the probabilistic nature of atmospheric phenomena. However, this comes at a
signiﬁcant cost overhead during inference, due to the iterative nature of denoising. Lang et al. [22] use
a scoring rule based objective function instead [ 24], with the implied computational beneﬁts over the
di!usion approach. While e!ectively addressing blurring, both approaches lead to build-up of small-scale
noise, requiring an ad-hoc truncation strategy in the latter case to suppress it. This build-up can be a
precursor to blow-up in NWP models [ 25] and attaining stable spectra remains a key challenge.
Most of today’s leading ML weather models repurpose mature architectures such as transformers and graph-
neural-networks that were fundamentally developed for other scientiﬁc ML tasks [ 26, 27]. This pragmatic
approach enables competitive medium-range skill, as demonstrated by numerous models, with little to
disambiguate between them [ 17]. As going beyond medium-range forecasts requires additional properties
beyond medium-range skill, bespoke geometric approaches o!er a simple and elegant alternative. These
methods are faithful to the underlying geometry, its topology and the symmetries of the underlying physics.
However, bespoke methods come with signiﬁcant engineering challenges, requiring custom implementations
and engineering frameworks to achieve the necessary scale of model training that is in turn needed to
achieve competitive skill [ 28].

### FourCastNet 3
We introduce FourCastNet 3 (FCN3), a skillful, probabilistic ML weather forecasting
system built as a hidden Markov model based on spherical signal processing primitives and a probabilistic
loss function in the spectral domain. Our method is purely convolutional, leveraging both local and global
spherical convolution kernels to better model the physical processes at various scales involved in weather
phenomena, while respecting spherical geometry and its inherent symmetries, enabling realistic ensemble
members.
FCN3 is trained end-to-end as an ensemble forecasting model, at scale. This approach retains the large
speed-ups o!ered by ML models, facilitating one-step generation of ensemble members, making it both
computationally e”cient and accurate. To enable training of a large FCN3 model, on a large hourly dataset
with multiple ensemble members, and multiple timesteps, we develop a hybrid machine learning paradigm
for simultaneous model- and data-parallelism, inspired by traditional NWP methods. The computational
domain is decomposed to simultaneously distribute both the model and the data during training. This is
combined with distributed batch- and ensemble-parallelism, resulting in extremely e”cient and scalable
training, which enabled seamlessly scaling training to over 1000 GPUs.

FCN3 outperforms the integrated forecasting system’s ensemble method (IFS-ENS) [ 29], the golden
standard for traditional NWP methods, and nearly matches the medium-range forecast skill of GenCast [ 6],
the leading probabilistic ML weather model, at double the temporal resolution. A single forecast of 15
days is computed in 60 seconds on a single NVIDIA H100 GPU - a speedup of 8x over GenCast and 60x
over IFS-ENS. Simultaneously, it o!ers the key beneﬁt of retaining stable predictions and accurate spectra
well into the subseasonal range with lead times of up to 60 days. This key achievement mitigates the
issue of blurring and addresses the issue of build-up of small-scale noise. The probabilistic skill, stability,
spectral ﬁdelity and low inference cost make FCN3 an interesting model with the potential of generating
large ensembles with potential applications spanning medium-range to subseasonal forecasting.

## 2 Probabilistic forecasts with hidden Markov models
FourCastNet 3 (FCN3) is formulated as a probabilistic model to address the chaotic nature of atmospheric
phenomena. Given the current atmospheric state un on a 0 .25→ grid at a time tn, it predicts the state
un+1 = Fω (un,t n,z n) at the next time step tn+1, 6 hours into the future. Stochasticity is introduced
through a hidden Markov model approach, where the model takes an extra conditioning input zn -a
random noise vector drawn from a number of spherical di!usion processes with di!erent length- and
timescales [ 30]. Figure 1 depicts this setup, and a detailed description is found in Appendix A.
The parameters ω of the model Fω are optimized with the aim of accurately approximating atmospheric
processes and matching the observed spatio-temporal distributions of physical variables. FCN3 uses an
end-to-end ensemble training approach, minimizing a composite probabilistic loss function (48) based
on the continuously ranked probability score (CRPS) (47). This objective compares the predictive
ensemble of marginals to ground-truth observations. Although training with the CRPS objective has been
shown to produce models with high predictive skill, these models have not generated ensemble members
with physically accurate spectra that correctly capture spatial correlations [ 16, 20, 22]. Although the
scalar-valued Continuous Ranked Probability Score (CRPS) (40) is a proper scoring rule – meaning it is
uniquely minimized when the predictive distribution matches the target distribution – this property does
not extend to summary scores that aggregate individual CRPS values across marginals, as is commonly
done when forecasting spatial or multivariate variables. This is particularly problematic for multi-variate
spatial processes, where the CRPS can be minimized in a point-wise manner by an unphysical ensemble.
To address this issue, we combine the spatial, point-wise CRPS loss term with a loss term in the spectral
domain. A similar approach using a low-pass ﬁltered spectral loss term has previously been adopted
by Kochkov et al. [16], but failed to accurately capture the high-frequency behavior of the underlying
processes. Our approach weights spectral coe”cients according to their multiplicity and enforces a good
match of the their distributions across all wavelengths. A detailed discussion of the objective function and
its motivation are provided in Appendix E.1.

## 3 Spherical neural operator architecture
Although a combined spectral and spatial probabilistic loss function encourages the learned operator to
be accurately represented across scales, the concrete parameterization is equally important in determining
the space of learnable operators and therefore their properties. As such, we choose a geometric approach
grounded in signal processing principles and symmetry considerations:
FCN3 is a spherical neural operator architecture and relies heavily on local and global spherical group
convolutions. More precisely, global convolution ﬁlters are parameterized in the spectral domain by
leveraging the convolution theorem on the sphere and the associated spherical harmonic transform (SHT)
[10]. This approach resembles classical pseudo-spectral methods such as IFS, which compute the PDE
operator in the spectral domain. Additionally, we employ spherical group convolutions 1 with learnable,
locally supported kernels. This is implemented using the framework for discrete-continuous (DISCO)
convolutions on the sphere [ 31, 32], which formulate the convolution in the continuous domain and
approximate the integral with a quadrature rule. This formulation enables anisotropic ﬁlters that are
better suited to approximate atmospheric phenomena such as adiabatic ﬂow conﬁned to vertically tilted
isentropes with characteristic morphology, or blocked ﬂow around topographic features. The localized
convolutional approach also resembles ﬁnite di!erencing - another building block encountered in most
classical NWP models.
Building on these convolutional principles, the overall FCN3 architecture is organized into three main
components: an encoder, a processor composed of several spherical neural operator blocks, and a decoder
(see Figure 1). These blocks adopt the structure of the popular ConvNeXt architecture [ 33], which
contain a convolution, a GeLU activation function [ 34], a point-wise multi-layer perceptron (MLP) and
1Group convolutions are convolutions formulated w.r.t. a symmetry group. For the two-dimensional sphere,
this is the rotation group of three-dimensional rotations SO(3).
a skip connection. We deliberately omit layer normalization, motivated by the importance of absolute
magnitudes in physical processes. The convolution ﬁlters are either parameterized in the spectral domain
or as approximately spherically equivariant local convolutions [ 10, 32]. In the latter case, we choose
smooth localized ﬁlter functions, parameterized by linear combinations of Morlet wavelets on a disk.
Through experimentation, we ﬁnd that a ratio of four local blocks to one global block yields the best
forecast skill. The encoder layer is comprised of a single local spherical convolutions and down-samples
the 721 → 1440 input/output signals to a latent representation on a 360 → 720 Gaussian grid with an
embedding dimension of 641. The decoder uses a combination of bilinear spherical interpolation and local
spherical convolution to up-sample the latent signals to the native resolution while mitigating aliasing
errors. Both encoder and decoder encode do not perform any channel mixing and instead encode input
signals separately, to avoid the mixing of signals with vastly di!erent spectral properties. Finally, water
channels are passed through a smooth, spline-based output activation function which constrains them to
positive values, while reducing the amount of high-frequency noise introduced through the non-linearity.
In contrast to most ML weather models which predict tendencies, i.e. the di!erence between the prediction
and the input, FCN3 predicts the next state directly. Empirically, we ﬁnd that this approach works better
in avoiding the build-up of high-frequency artifacts. Moreover, predicting tendencies may be interpreted as
restricting the model to Euler time-stepping, which may adversely a!ect the space of learnable operators
[22]. A detailed account of signal-processing considerations on the sphere and our ﬁlter parameterizations
is provided in Appendix B. Furthermore, architectural choices and hyperparameters are discussed in detail
in Appendix C.

## 4 Scalable training through hybrid parallelism
Training models with large internal representations such as FCN3 requires more memory than what is
available on a single GPU for their forward and backward passes. This memory requirement is further
exacerbated by autoregressive rollouts, where multiple forward and backward passes need to be ﬁt into
GPU memory. These considerations limit the size of the model to the memory available per GPU, and thus
set the maximum scale for most models. While some models such as GraphCast use gradient checkpoint
to enable the memory-intensive training [ 4], this comes with the signiﬁcant downside of trading memory
for compute, increasing already long iteration times further in training.
By distributing models across multiple GPUs, model parallelism o!ers an alternative path for practitioners
to reduce the memory requirements and train much bigger models. This approach greatly improved the
ﬁdelity and performance of modern ML models [ 35–37] and is the foundation of the success of current large
language models (LLM) such as ChatGPT 4 [ 38], Llama 3 [ 39], and others. Neural networks generally
scale well with available data and oftentimes, training larger models comes with an increase in skill, as
long as more training data is available [ 28]. This creates a unique challenge for scientiﬁc ML methods,
where the training data is often high-dimensional, in comparison to language modeling or computer vision
tasks. In the case of FCN3, a typical sample at 0 .25→ resolution consists of 721 → 1440 ﬂoating points per
variable, and multiple tens of variables are normally used for skillful predictions. This renders ML driven
weather prediction considerably more data-intensive than many other ML tasks.
Model parallelism is inspired by classical numerical methods, where not only the model and weights are
split across ranks, but also the data which the model processes. Model parallelism is typically achieved
through feature-space parallelism, i.e. by splitting the feature maps across multiple GPU. This approach
is heavily used in modern distributed LLMs, alongside other parallelism paradigms such as pipeline- and
traditional batch-parallelism. To enable the training of FCN3, we implement spatial model parallelism
(also referred to as domain parallelism), where both the model and data are split across ranks by employing
a spatial domain decomposition. This approach is inspired by traditional distributed scientiﬁc computing
applications and requires the implementation of distributed variants of all spatial algorithms (see Figure 2).
Besides these two approaches as well as traditional batch parallelism, another approach is to split members
of the same forecasting ensemble across multiple GPU. This variant of data parallelism is highly e”cient
because di!erent ensemble members are computationally independent until the loss computation, which
usually requires some communication across the ensemble group of GPUs.
The training of FCN3 requires spatial model parallelism via domain decomposition as well as ensemble
and batch parallelism. We will refer to the former as model and the latter two as data parallelism. We
have implemented all of these features in Makani, a framework for large-scale distributed training of ML
based weather models. For a more detailed description of parallelization features, cf. section G.
This paradigm enables us to train large principled models by scaling training to thousands of GPUs and
more. FCN3 is trained on historic atmospheric ERA5 reanalysis data ranging from 1980 to 2016. ERA5
is a multi-decadal, self-consistent record and represents our best understanding of Earth’s atmospheric
system [ 40]. Training is split into stages, thereby forming a curriculum training approach. The initial
pre-training phase focuses on the model’s 6-hourly prediction skill, by utilizing all hourly samples from the
ERA5 training dataset, constructing 6 hour lead time input-target-pairs that start at each discrete UTC
hour. The model is trained for 208,320 gradient descent steps on this dataset with a batch size of 16 and
an ensemble size of 16. This initial training stage was carried out on 1024 NVIDIA H100 on the NVIDIA
Eos Supercomputer for a total of 78 hours. In the second pre-training phase, the model is trained on
6-hourly initial conditions using 4 autoregressive rollout steps. This is performed for 5,040 steps while
lowering the learning rate every 840 steps. The second pre-training stage took 15 hours on 512 NVIDIA
A100 GPUs to complete and was carried out on the NERSC Perlmutter system. The ﬁnal is ﬁne-tuned on
6-hourly samples ranging from 2012 to 2016 to account for potential drifts in the distribution and improve
performance on data that lie in the near- to medium-term future. This ﬁnal stage is carried out on 256
NVIDIA H100 GPUs on the Eos system and took 8 hours to complete. As a single model instance does
not ﬁt on a 80Gb VRAM GPU, we leverage the previously described spatial parallelism, splitting the
data and the model. This ranges from a 4-fold split in pretraining to a 16-fold split during ﬁnetuning, due
to the increased memory requirements from autoregressive training. Details of the training methodology
and setup are outlined in Appendix E.

## 5 Results
Key performance scores of FCN3 such as continuously ranked probability score (CRPS) and ensemble-mean
RMSE are averaged over 12-hourly initial conditions in the out-of-sample year 2020 and reported in
Figure 3. FCN3 beats the gold-standard physics-based NWP model IFS-ENS by a margin that is virtually
indistinguishable from GenCast, the state-of-the-art data-driven weather model. Our approach enables
direct, one-step generation of ensemble members and can generate a single 15-day forecast at a temporal
resolution of 6 hours and a spatial resolution of 0 .25→ in a matter of 60 seconds on a single NVIDIA H100
GPU. In comparison, a 15-day forecast of GenCast takes 8 minutes on a Cloud TPU v5 instance (at half
the temporal resolution) [ 6], and an IFS forecast takes about one hour on 96 AMD Epyc Rome CPUs (at
9km operational resolution) [ 41]. Barring the di!erences in hardware and resolution, this constitutes a
speed-up of ↑8x over GenCast and a speed-up of ↑60x over IFS-ENS.
Crucially, the 50-member FCN3 ensemble forecast is well-calibrated with spread-skill ratios approaching
1, indicating interchangeability between observations and ensemble members in the forecast. This is
conﬁrmed via rank-histograms, which report the frequencies of the ordinal ranks of the observation within
the predictive ensemble. The temporal evolution of the rank histograms closely mirrors the spread-skill
ratios, indicating a slightly over-dispersive ensemble at short lead times of up to 24 hours, which then
becomes under-dispersive and then gradually relaxes to a ﬂat rank-histogram. These results are especially
encouraging, given that the evaluated 50 member ensemble is larger than the 16 ensemble members used
in training, indicating that even larger ensembles are justiﬁable to test during inference.
It is important to investigate case studies, since scores such as ensemble-mean RMSE and CRPS are
incomplete metrics that alone do not provide a comprehensive view of a probabilistic weather forecast.
For example, the CRPS score only evaluates the accuracy of the predictive distribution point-wise and
does not take tempo-spatial correlations into account. As such, a perfect forecast from the ground-truth
distribution, which is scrambled by shu#ing the ensemble members at each point will result in unphysical
predictions yet still retain the optimal CRPS score. Similarly, RMSE scores can be easily improved by
blurring forecasts, rendering them useless for all practical purposes. A key challenge in data-driven weather
models is to reproduce the physical ﬁdelity of traditional NWP models and reduce possible spurious
correlations that stem from the data-driven approach.
Figure 4 examines a case study showing wind intensities at 850hPa and geopotential height at 500hPa of a
FCN3 forecast initialized on 2020-02-11 at 00:00:00 UTC, 48 hours before the extra-tropical storm Dennis
made its landfall over Ireland and the British Isles. The close-up plots in Figure 4 indicate that FCN3 is
capable of faithfully simulating this event, reproducing both realistic wind intensities and appropriate
co-variation of ﬂow with the pressure ﬁeld. This is conﬁrmed in the angular power spectral density (PSD)
of the 500hPa geopotential height, reported in the bottom row. FCN3 retains perfectly the correct slopes
in the power spectra, a desirable property towards better ML weather models with high physical ﬁdelity.
Even at long lead times of 30 days, we observe no apparent degradation of the angular power spectra and
predictions retain their e!ective resolution, remaining sharp even at long lead times.
The spectral ﬁdelity of FCN3 is also observed in Figure 5, which depicts power spectral densities averaged
over the entire evaluation year of 2020 and the respective relative error w.r.t. the angular power spectrum
of the ERA5 ground truth. Even at high wavenumbers, we observe that the relative error remains bounded
with deviations ranging from ↓0.2t o0 .2.
We postulate that the spectral properties are a result of our careful architectural design choices, which
reﬂect geometrical and signal-processing principles, and the combined CRPS loss function which enforces
the correct local and global distribution, thus encouraging the model to learn the correct spatial correlations.
Competing, deterministic ML weather models typically display a decay of high-frequency information,
which appears as blurring. Even the CRPS-trained hybrid weather model NeuralGCM shows signiﬁcant
blurring in high-frequency modes. Moreover, newer, probabilistic ML weather models such as GenCast
[6] and AIFS-CRPS [ 22] cannot faithfully retain the correct spectral signatures and show a build-up of
high-frequency modes, as illustrated in Figure 4. In traditional NWP models, such build-ups can be a
precursor to an imminent blow-up [ 25]. As such, this constitutes a major milestone towards physically
faithful data-driven, probabilistic weather models, which can be e”ciently evaluated even at longer lead
times.
Additional evaluation of FCN3, angular and zonal power spectral densities, alongside physical consistency
tests, are provided in Appendix F. The detailed evaluation conﬁrms that FCN3 is a probabilistically
skillful, computationally e”cient global weather model, with unprecedented spectral ﬁdelity and a high
degree of physical realism. Forecasts remain stable well into the subseasonal range of 60 days, thus paving
the way toward subseasonal forecasts and large ensembles at these lead times.

## 6 Conclusions
We present FourCastNet 3 (FCN3), a novel probabilistic weather forecasting model that leverages
spherical signal processing and a hidden-Markov ensemble formulation, trained end-to-end with a
probabilistic objective in both spectral and spatial domains. FCN3 achieves skillful and computationally
e”cient forecasts, outperforming traditional numerical weather prediction (NWP) methods and matching
the performance of state-of-the-art di!usion models at a fraction of the computational cost. This is
accomplished using a purely convolutional architecture based on spherical group convolutions, in contrast
to the prevailing transformer-based approaches. Notably, FCN3 generates physically realistic spectra across
all wavelengths up to the cuto! in the training data, avoiding the overly smooth or spurious high-frequency
artifacts that challenge other machine learning models. This ﬁdelity enables stable, sharp forecasts even
at extended lead times of up to 60 days, positioning FCN3 as a promising tool for subseasonal prediction
with large ensembles.
FCN3 introduces major computational and practical improvements that make large-scale, high-resolution
ensemble forecasting more accessible than ever. Its massively parallel training workﬂows, model and
ensemble parallelism, and low inference cost enable rapid, e”cient production of large ensemble forecasts.
In-situ diagnostics and scoring can be performed during model execution, eliminating the need to store
terabytes of data and removing storage and I/O bottlenecks that have historically limited ensemble
analysis. All key components, including training and inference code, are fully open-source, providing the
research community with transparent, reproducible tools for both operational and experimental weather
prediction.
As an ensemble model, FCN3 enables detailed exploration of multiple plausible future weather scenarios
from a single initialization, making it a powerful tool for studying atmospheric dynamics, predictability, and
the statistics of low-probability, high-impact events. Looking ahead, we plan to extend FCN3 to include
precipitation as a diagnostic output and to integrate data assimilation uncertainty, further broadening its
applicability and impact. Together, these innovations position FCN3 as a robust, e”cient, and extensible
foundation for next-generation probabilistic weather forecasting and atmospheric science research.

## Data and materials availability
FourCastNet 3’s training code is available in Makani, a training framework used for scale training of
ML weather models to 1000s of GPUs. It is openly available at https://github.com/NVIDIA/makani
under the Apache License 2.0. The ERA5 training data is openly available at https://cds.climate.
copernicus.eu/datasets/reanalysis-era5-single-levels . Finally, torch-harmonics, our library
for machine-learning and di!erentiable signal processing on the sphere, is openly available at https:
//github.com/NVIDIA/torch-harmonics under the BSD-3-Clause license.