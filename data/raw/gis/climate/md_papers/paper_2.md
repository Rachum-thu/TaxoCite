# FuXi: A cascade machine learning forecasting system for 15-day global weather forecast

## Abstract
Over the past few years, the rapid development of machine learning (ML)
models for weather forecasting has led to state-of-the-art ML models
that have superior performance compared to the European Centre for
Medium-Range Weather Forecasts (ECMWF)’s high-resolution forecast
(HRES), which is widely considered as the world’s best physics-based
weather forecasting system. Speciﬁcally, ML models have outperformed
HRES in 10-day forecasts with a spatial resolution of 0.25 °. However,
the challenge remains in mitigating accumulation of forecast errors for
longer e!ective forecasts, such as achieving comparable performance to
the ECMWF ensemble in 15-day forecasts. Despite various e!orts to
reduce accumulation errors, such as implementing autoregressive multi-
time step loss, relying on a single model has been found to be insu”cient
for achieving optimal performance in both short and long lead times.
Therefore, we present FuXi, a cascaded ML weather forecasting system
that provides 15-day global forecasts at a temporal resolution of 6 hours
and a spatial resolution of 0.25 °. FuXi is developed using 39 years of the
ECMWF ERA5 reanalysis dataset. The performance evaluation demon-
strates that FuXi has forecast performance comparable to ECMWF
ensemble mean (EM) in 15-day forecasts. FuXi surpasses the skillful fore-
cast lead time achieved by ECMWF HRES by extending the lead time for
Z500 from 9.25 to 10.5 days and for T 2M from 10 to 14.5 days. More-
over, the FuXi ensemble is created by perturbing initial conditions and
model parameters, enabling it to provide forecast uncertainty and demon-
strating promising results when compared to the ECMWF ensemble.
Keywords: weather forecast, machine learning, accumulation error, cascade,
FuXi, transformer

## Introduction
Accurate weather forecasts play an important role in many aspects of human
society. Currently, national weather centers around the world generate weather
forecasts using numerical weather prediction (NWP) models, which simulate
the future state of the atmosphere. Nevertheless, running NWP models often
requires high-performance computing systems, with some simulations taking
several hours using thousands of nodes. The Integrated Forecast Systems (IFS)
of the European Centre for Medium-range Weather Forecast (ECMWF) is
widely regarded as the most accurate global weather forecast model [ 1]. The
ECMWF’s high-resolution forecast (HRES) runs at a horizontal resolution
of 0.1 °with 137 vertical levels for 10-day forecasts. However, uncertainty in
weather forecasts is inevitable due to the limited resolution, approximation
of physical processes in parameterizations, errors in initial conditions (and
boundary conditions for regional models), and the chaotic nature of the atmo-
sphere. Additionally, the degree of uncertainty and the magnitude of errors
in weather forecasts increases as forecast lead time. One way to address this
uncertainty is to run an ensemble of forecasts by incorporating perturbations
in initial conditions and physical parameterizations in the NWP model. The
ECMWF ensemble prediction system (EPS) [ 2] provides forecasts up to 15
days and is comprised of one control member and 50 perturbed members. The
IFS Cycle 48r1, which was introduced in June 2023, upgrade the spatial and
vertical resolution and vertical resolution of the EPS same as the HRES [ 3],
which was made possible by a new supercomputer with enhanced capacity.
Prior to the upgrade, the EPS ran at a lower spatial resolution of 18 km and
had fewer vertical levels of 91, due to the substantial computational demands
of running 51 members with limited computing resources.

In recent years, there have been increasing e!orts to replace the traditional
NWP models with machine learning (ML) models for weather forecasting [ 4].
ML-based weather forecasting systems have several advantages over NWP
models, including faster speeds and the potential to provide higher accuracy
than uncalibrated NWP models due to training with reanalysis data [ 4]. To
facilitate intercomparison between di!erent ML models, the WeatherBench
benchmark was introduced to evaluate medium-range weather forecasting (i.e.,
3-5 days) [ 5, 6]. WeatherBench was created by regridding ERA5 reanalysis
data [7] from 0.25→ resolution to three di!erent resolutions (5 .625→,2 .8125→ and
1.40625→). Several studies have aimed to improve forecast performance on this
dataset [ 8–10]. For example, Rasp et al. [ 8] used a deep residual convolutional
neural network (CNN) known as ResNet [ 11] to predict 500 hPa geopoten-
tial ( Z500), 850 hPa temperature ( T 850), 2-meter temperature ( T 2M ), and
total precipitation ( TP ) at a spatial resolution of 5 .625→ for up to 5 days.
They found that the ResNet model has similar performance compared to phys-
ical baseline models, such as IFS T42 and T63, with a comparable resolution.
Meanwhile, Hu et al. [ 10] proposed the SwinVRNN model, which utilizes a
Swin Transformer-based recurrent neural network (RNN) (SwinRNN) model
coupled with a perturbation module to learn multivariate Gaussian distribu-
tions based on the Variational Auto-Encoder framework. They demonstrated
the SwinVRNN model’s potential as a powerful ML-based ensemble weather
forecasting system, with good ensemble spread and better accuracy compared
to IFS in terms of T 2M and 6-hourly TP in 5-day forecasts with a 5 .625→
resolution.

While ML models have shown good performance in weather forecasting,
their practical values are limited because of their forecasts’ low resolution (e.g.,
5.625→). As a remarkable breakthrough, the FourCastNet model [ 12]i st h eﬁ r s t
of its kind to provide high-resolution global weather forecasts of 0 .25→ for a time
period of 7 days. It integrates the Adaptive Fourier neural operator (AFNO)
[13] with a Vision Transformer (ViT) [ 14]. However, FourCastNet’s forecast
accuracy is still worse than HRES’s. SwinRDM [ 15] distinguishes itself as the
ﬁrst ML-based weather forecasting system to outperform ECMWF HRES in
5-day forecasts at a spatial resolution of 0 .25→. SwinRDM integrates Swin-
RNN+, an improved version of SwinRNN that surpasses ECMWF HRES at
a spatial resolution of 1 .40625→, with a di!usion-based super-resolution model
that increases the resolution to 0 .25→. Pangu-Weather [ 16] shows its superior
performance compared to ECMWF HRES in 7-days forecasts at a resolution
of 0 .25→. Additionally, GraphCast [ 17], an autoregressive model that imple-
ments a graph neural network (GNN), outperforms HRES in 90% of the 2760
variable and lead time combinations in 10-day forecasts.

Although ML models have shown promising results in generating weather
forecasts for 10 days, long-term forecasting remains challenging due to cumu-
lative errors. The iterative forecasting method, which uses the model outputs
as inputs for subsequent predictions, is a commonly used approach in devel-
oping ML-based weather forecasting systems. This approach is similar to the
time-stepping methods used in conventional NWP models [ 18]. However, as the
number of iterations increases, errors in the model outputs accumulate, which
may lead to signiﬁcant discrepancies with the training data and unrealistic
values in long-term forecasts. Many research has been conducted to enhance
the stability and accuracy of long-term forecasts. Weyn et al. [ 9] proposed a
multi-time-step loss function to minimize errors over multiple iterated time
steps. Rasp et al. [ 5] compared iterative forecasts with direct forecasts that
predict speciﬁc lead times and found the latter to be more accurate. However,
one limitation of direct forecasts is that separate models need to be trained for
each lead time. The FourCastNet [ 12] model underwent two training phases:
pre-training, in which the model is optimized to map one time step to the next
with a 6-hour interval, and ﬁne-tuning to minimize errors in two-step predic-
tion, similar to the multi-time-step loss function proposed by Weyn et al. On
the other hand, Bi et al. proposed a hierarchical temporal aggregation strat-
egy for Pangu-Weather’s forecasts, training four separate models for 1-hour,
3-hour, 6-hour, and 24-hour forecasts [ 16]. They demonstrated that running
the 24-hour model 7 times is better than running the 1-hour model 168 times
as it signiﬁcantly reduces the accumulation errors for 7-day forecasts. How-
ever, they acknowledged that training a model directly predicting the lead time
beyond 24 hours is challenging with their current model. Meanwhile, Lam et
al. employed a curriculum training schedule following pre-training to improve
GraphCast’s ability to make accurate forecasts for multiple steps [ 17]. Increas-
ing autoregressive steps results in excessive memory and computational costs,
thereby limiting the maximum feasible number of steps. Chen et al. [ 19] pro-
posed a reply bu!er mechanism to mimic the long-lead autoregressive forecasts
with improved computational e”ciency and reduced memory costs. The study
by Lam et al. [ 17] revealed that GraphCast’s performance decreases in short
lead times and improves at longer lead times as the number of autoregressive
steps increases. Thus, using a single model is insu”cient for achieving the best
performance for both short and long lead times.

To conclude, signiﬁcant progress have been achieved in ML-based weather
forecasting, particularly in 10-day forecasts where the ML models have out-
performed ECMWF HRES. However, further breakthroughs are necessary to
address the issues related to iterative accumulated errors and enhance the
accuracy of forecasts for longer lead times. The next signiﬁcant goals are to
achieve comparable performance to ECMWF ensemble, of which the ensemble
mean (EM) often has greater skill than the deterministic forecasts for longer
lead times, and to increase the forecast lead time beyond 10 days. The objec-
tive of this study is to reduce the accumulation error and generate ML-based
weather forecasts for 15 days that have performance comparable to ECMWF
EM. However, since a single model has been shown to be incapable of achieving
optimal forecast performance across various forecast lead times, we propose
a novel cascade ML model architecture for weather forecasting based on pre-
trained models, each optimized for speciﬁc forecast time windows. As a result,
we present FuXi 1 weather forecasting system that generates 15-day forecasts
at the spatial resolution of 0 .25→. FuXi is a cascade of models optimized for
three sequential forecast time periods of 0-5 days, 5-10 days, and 10-15 days,
respectively. The base FuXi model is an autoregressive model designed to e”-
ciently extract complex features and learn relationships from a large volume
of high-dimensional weather data. Speciﬁcally, 39 years of 6-hourly ECMWF
ERA5 reanalysis data at a spatial resolution of 0 .25→ are used for developing
the FuXi system. The evaluation shows that FuXi signiﬁcantly outperforms
ECMWF HRES and achieves comparable performance to ECMWF EM for
the ﬁrst time. FuXi extends the skillful forecast lead time, as indicated by
whether anomaly correlation coe”cient (ACC) being greater than 0.6, to 10.5
and 14.5 days for Z500 and T 2M , respectively. Moreover, ensemble forecasts
provide greater values beyond EM by o!ering estimates of forecast uncertainty
and enabling skillful predictions for longer lead times. Therefore, we developed
the FuXi ensemble forecast by introducing perturbations to initial conditions
and model parameters in order to generate ensemble forecasts. The evaluation
based on the continuous ranked probability score (CRPS) demonstrates that
the FuXi ensemble performs comparably to the ECMWF ensemble within a
forecast lead time of 9 days for Z500, T 850, mean sea-level pressure ( MSL ),
and T 2M .

Overall, our contribution to this work can be summarized as follows:
• We propose a novel cascade ML model architecture for weather forecasting,
which aims to reduce accumulation errors.
• FuXi achieves comparable performance to ECMWF EM and extends the
skillful forecast lead time (ACC >0.6) to 10.5 and 14.5 days for Z500 and
T 2M , respectively.

## Dataset

### ERA5
ERA5 is the ﬁfth generation of the ECMWF reanalysis dataset, providing
hourly data of surface and upper-air parameters at a horizontal resolution of
approximately 31 km and 137 model levels from January 1940 to the present
day [ 7]. The dataset is generated by assimilating high-quality and abun-
dant global observations using ECMWF’s IFS model. Given its coverage and
accuracy, the ERA5 data is widely regarded as the most comprehensive and
accurate reanalysis archive. Therefore, we use the ERA5 reanalysis dataset as
the ground truth for the model training.

We use a subset of the ERA5 dataset spanning 39 years, which has a
spatial resolution of 0 .25→ (721 → 1440 latitude-longitude grid points) and a
temporal resolution of 6 hours. In this work, we focus on predicting 5 upper-air
atmospheric variables at 13 pressure levels (50, 100, 150, 200, 250, 300, 400,
500, 600, 700, 850, 925, and 1000 hPa), and 5 surface variables. The 5 upper-air
atmospheric variables are geopotential ( Z), temperature ( T ), u component of
wind ( U ), v component of wind ( V ), and relative humidity ( R). Additionally,
5 surface variables are T 2M , 10-meter u wind component ( U 10), 10-meter v
wind component ( V 10), MSL , and TP 2. In total, 70 variables are predicted
and evaluated.

Following previous studies in splitting the data into training, validation,
and testing set [ 12, 17], the training set consists of 54020 samples spanning
from 1979 to 2015. The validation set contains 2920 samples corresponding to
the years 2016 and 2017, while out-of-sample testing is performed using 1460
samples from 2018.

### HRES-fc0 and ENS-fc0 dataset
In this study, we evaluate our model against the ERA5 reanalysis data. Besides,
we also created two reference datasets, HRES-fc0 and ENS-fc0, which consist
of the ﬁrst time step of each HRES and ensemble control forecast, respectively.
We use these datasets to assess the performance of ECMWF HRES and EM.
This approach aligns with that used by Haiden et al. [ 1] and Lam et al. [ 17]
in evaluating ECMWF forecasts.

## Methodology

### Generating 15-day forecasts using FuXi
The FuXi model is an autoregressive model that leverages weather parameters
(Xt↑1,X t) from two previous time steps as input to forecast weather parame-
ters at the upcoming time step ( Xt+1). t, t ↑ 1, and t +1 represent the current,
the prior, and upcoming time steps, respectively. The time step considered in
this model is 6 hours. By utilizing the model’s outputs as inputs, the system
can generate forecasts with di!erent lead times.

Generating 15-day forecasts using a single FuXi model requires 60 iterative
runs. Pure data-driven ML models, unlike physics-based NWP models, lack
physical constraints, which can result in signiﬁcantly growing errors and unre-
alistic predictions for long-term forecasts. Using an autoregressive, multi-step
loss e!ectively minimizes accumulation error for long lead times [ 17]. This loss
is similar to the cost function applied in the four-dimensional variational data
assimilation (4D-Var) method, which aims to identify the initial weather con-
ditions that optimally ﬁt observations distributed over an assimilation time
window. Although increasing the autoregressive steps leads to more accurate
forecasts for longer lead times, it also results in less accurate results for shorter
lead times. Besides, increasing autoregressive steps require more memory and
computing resources for handling gradients during the training process, similar
to increasing the assimilation time window of 4D-Var.

When making iterative forecasts, error accumulation is inevitable as lead
times increase. Also, previous studies indicate that a single model can not
perform optimally across all lead times. To optimize performance for both
short and long lead times, we propose a cascade [ 20, 21] model architecture
using pre-trained FuXi models, ﬁne-tuned for optimal performance in speciﬁc
5-day forecast time windows. These windows are referred to as FuXi-Short (0-
5 days), FuXi Medium (5-10 days), and FuXi-Long (10-15 days). FuXi-Short and FuXi Medium outputs from the 20th and 40th steps
are used as inputs to FuXi-Medium and FuXi-Long, respectively. Unlike the
greedy hierarchical temporal aggregation strategy employed in Pangu-Weather
[16], which utilizes 4 models with forecast lead times of 1 h, 3 h, 6 h, and 24
h to minimize the number of steps, the cascaded FuXi model does not su!er
from temporal inconsistency. The cascaded FuXi model performs comparably
to ECMWF EM in 15-day forecasts.

### FuXi model architecture
The model architecture of the base FuXi model consists of three main compo-
nents: cube embedding, U-Transformer, and a fully connected (FC) layer. The input data combines both upper-air and surface variables and creates a data cube with dimensions of 2 → 70 → 721 → 1440, where 2, 70, 721, and 1440 represent the two preceding time steps ( t ↑ 1 and t), the total number of input variables, latitude ( H) and longitude ( W ) grid points, respectively.

Firstly, the high-dimensional input data undergoes dimension reduction to C →180→360 through joint space-time cube embedding, where C is the number of channels, and is set to be 1536. The primary purpose of cube embedding is to reduce the temporal and spatial dimensions of input data, making it less redundant. Subsequently, the U-Transformer processes the embedded data, and prediction follows using a simple FC layer. The output is initially reshaped to 70 → 720 → 1440, then restored to the original input shape of 70 → 721 → 1440 by bilinear interpolation. The following subsections provide details for each component in the base FuXi model.

#### Cube embedding
To reduce the spatial and temporal dimensions of input and accelerate the
training process, the space-time cube-embedding [ 22] is applied. A similar
approach, patch embedding, which divides an image into N → N patches
with each patch being transformed into a feature vector, was used in the
Pangu-Weather model [ 16]. The cube embedding applies a 3-dimensional (3D)
convolution layer, with a kernel and stride of 2 →4→4 (equivalent to T
2 → H
4 → W
4 ), and output channels numbering C. Following cube embedding, a layer normalization (LayerNorm) [ 23] is utilized to improve training stability. The result is a data cube with dimensions of C → 180 → 360.

#### U-Transformer
This subsection presents the design of the U-Transformer.

Recently, the ViT [ 14] and its variants have demonstrated remarkable
performance in various computer vision tasks by using the multi-head self-
attention, which enables the simultaneous processing of sequential input data.
Nevertheless, global self-attention is infeasible for processing high-resolution
inputs due to its quadratic computational and memory complexity with
respect to the input size. Swin Transformer was proposed as a solution [ 24]
to improve computational e”ciency by limiting computation of self-attention
only within the non-overlapping local windows. Besides, the shifted-window
mechanism allows for cross-connections between windows. As a result, the Swin
Transformer has shown superior performance on various benchmarks and is
frequently used as a backbone architecture in many vision tasks. Additionally,
many researchers have developed ML-based weather forecasting models using
Swin Transformer blocks [ 10, 12, 15, 16].

However, training and applying a large-scale Swin Transformer model
for high-resolution inputs reveals several issues, including training instabil-
ity. To address these issues, Swin Transformer V2 [ 25] was proposed, which
upgrades the original Swin-Transformer (V1) [ 24] by using the residual post-
normalization instead of pre-normalization, scaled cosine attention instead
of the original dot product self-attention, and log-spaced coordinates instead
of the previous linear-spaced coordinates. As a result, Swin Transformer V2
has 3 billion parameters and advances state-of-the-art performance on multiple
vision task benchmarks.

The U-Transformer is constructed using 48 repeated Swin Transformer V2
blocks and calculates the scaled cosine attention as follows:
Attention(Q, K, V) = (cos (Q, K) /ω + B)V (1)
where B represents the relative position bias and ω is a learnable scalar,
which is not shared across heads and layers. The cosine function is naturally
normalized, which leads to smaller attention values.

The U-Transformer also includes a downsampling and upsampling block from
the U-Net model [ 26]. The downsampling block, referred to as the Down Block,
reduces the data dimension to C → 90 → 180, thereby minimizing computational and memory requirements for self-attention calculation. The Down Block consists of a 3 → 3 2-dimensional (2D) convolution layer with a stride of 2, and a residual [ 27] block that has two 3 → 3 convolution layers followed by a group normalization (GN) layer [ 28] and a sigmoid-weighted linear unit (SiLU) activation [ 29, 30]. The SiLU activation is calculated by multiplying the sigmoid function with its input ( ε(x)→x). The upsampling block, known as Up Block, has the same residual block as used in the Down Block, along with a 2D transposed convolution [ 31] with a kernel of 2 and a stride of 2. The Up Block scales the data size back up to C → 180 → 360. Furthermore, a skip connection is included that concatenates the outputs from the Down Block with those of the transformer blocks before being fed into the Up Block.

### FuXi model training
This section outlines the training process for FuXi models. The training proce-
dure involves two steps: pre-training and ﬁne-tuning, similar to the approach
used for training GraphCast [ 17].

#### One-step Pre-training
The pre-training step involves supervised training and optimizing the FuXi
model to predict a single time step using the training dataset. The loss function
used is the latitude-weighted L1 loss, which is deﬁned as follows:
L1= 1
C → H → W
C∑
c=1
H∑
i=1
W∑
j=1
ai | ˆXt+1
c,i,j ↑ Xt+1
c,i,j | (2)
where C, H, and W are the number of channels and the number of grid points
in latitude and longitude direction, respectively. c, i, and j are the indices
for variables, latitude and longitude coordinates, respectively. ˆXt+1
c,i,j and Xt+1
c,i,j
are predicted and ground truth for some variable and locations (latitude and
longitude coordinates) at time step of t + 1. ai represents the weight at latitude
i and the value of ai decreases as latitude increases. The L1 loss is averaged
over all the grid points and variables.

The FuXi model is developed using the Pytorch framework [ 32]. Pre-
training of the model requires approximately 30 hours on a cluster of 8 Nvidia
A100 GPUs. The model is trained with 40000 iterations using a batch size of 1
on each GPU. The AdamW [ 33, 34] optimizer is used with parameters ϑ1=0.9
and ϑ2=0.95, an initial learning rate of 2.5 →10↑4, and a weight decay coe”-
cient of 0.1. Scheduled DropPath [ 35] with a dropping ratio of 0.2 is employed
to prevent overﬁtting. In addition, Fully-Sharded Data Parallel (FSDP) [ 36],
bﬂoat16 ﬂoating point precision, and gradient check-pointing [ 37] are applied
to reduce memory costs during model training.

#### Fine-tuning cascaded models
After pre-training, the base FuXi model is ﬁrst ﬁne-tuned for optimal perfor-
mance for 6-hourly forecasts spanning from 0 to 5 days (0-20 time steps). This
ﬁne-tuning process is performed using an autoregressive training regime and
curriculum training schedule to increase the number of autoregressive steps
from 2 to 12, following the ﬁne-tuning approach of the GraphCast model [ 17].
This ﬁne-tuned model is referred to as FuXi-Short. With weights from FuXi-Short, the FuXi-Medium model is initialized and then ﬁne-tuned
for optimal forecast performance for 5 to 10 days (21-40 time steps). Implementing the online inference of FuXi-Short to get output at the 20th time step
(5th day), which is required for input to the FuXi-Medium model during its
ﬁne-tuning process, is inappropriate due to signiﬁcant memory consumption
and the slowdown of the ﬁne-tuning process for FuXi-Medium. To address this
issue, the results of FuXi-Short for six years of data (2012-2017) are cached on
a hard disk beforehand. The same procedure for ﬁne-tuning FuXi-Medium is
repeated for the ﬁne-tuning of FuXi-Long, optimized for generating forecasts
of 10-15 days. Finally, FuXi-Short, FuXi-Medium, and FuXi-Long are cascaded
to produce the complete 15-day forecasts. Cascade helps to reduce accumulation errors and improve forecast performance for longer lead times.

During the ﬁne-tuning process, the model was trained using a constant
learning rate of 1 →10↑7. It takes approximately two days to ﬁne-tune each of
the cascaded FuXi models on a cluster of 8 Nvidia A100 GPUs.

### FuXi ensemble forecast
Weather forecasting is inherently uncertain due to the chaotic nature of the
weather system [ 38]. To address this uncertainty, ensemble forecasting is nec-
essary, particularly for longer lead times. Additionally, since ML models can
generate forecasts at signiﬁcantly lower computational costs compared to con-
ventional NWP models, we generated a 50-member ensemble forecast using
the FuXi model. Following the approach used by ECMWF for ensemble runs,
which involves perturbing both initial conditions and model physics [ 39, 40],
we incorporated random Perlin noise [ 16] into the initial conditions and imple-
mented the Monte Carlo dropout (MC dropout, dropout rate is 0.2) [ 41] to
perturb the model parameters. More speciﬁcally, each of the 49 perturbations
contains 4 octaves of Perlin noise, a scaling factor of 0.5, and the number of
periods of noise to generate along each axis (channel, latitude, and longitude)
being 1, 6 and 6, respectively.

### Evaluation method
We follow [ 5] to evaluate forecast performance using latitude-weighted root
mean square error (RMSE) and ACC, which are calculated as follows:
RM SE(c, ω)= 1
| D |
∑
t0 ↔D
√ 1
H → W
H∑
i=1
W∑
j=1
ai( ˆXt0 +ω
c,i,j ↑ Xt0 +ω
c,i,j )
2
(3)
ACC (c, ω)= 1
| D |
∑
t0 ↔D
∑
i,j ai( ˆXt0 +ω
c,i,j ↑ M t0 +ω
c,i,j )( ˆXt0 +ω
c,i,j ↑ M t0 +ω
c,i,j )√∑
i,j ai( ˆXt0 +ω
c,i,j ↑ M t0 +ω
c,i,j )2 ∑
i,j ai( ˆXt0 +ω
c,i,j ↑ M t0 +ω
c,i,j )2
(4)
where t0 is the forecast initialization time in the testing set D, and ω is the
forecast lead time steps added to t0. M represents the climatological mean
calculated using ERA5 reanalysis data between 1993 and 2016. Additionally,
to improve the discrimination of the forecast performance among models with
small di!erences, we use the normalized RMSE di!erence between model A and
baseline B calculated as (RM SEA ↑ RM SEB )/RM SEB , and the normalized
ACC di!erence represented by (ACCA ↑ ACCB )/(1 ↑ ACCB ). Negative val-
ues in normalized RMSE di!erence and positive values in normalized ACC
di!erence indicate that model A performs better than the baseline model B.
To evaluate the performance of ECMWF HRES and EM, the veriﬁca-
tion method implemented by ECMWF [ 1] is used where the model analysis,
namely HRES-fc0 and ENS-fc0, serve as the ground truth for HRES and EM,
respectively.

In addition, we assess the quality of ensemble forecasts by calculating two
metrics: the CRPS [ 42, 43] and the spread-skill ratio (SSR). The CRPS is
computed using the following equation:
CRPS =
∫ ↗
↑↗
[F ( ˆXt0 +ω
c,i,j ) ↑H (Xt0 +ω
c,i,j ↓ z)] dz (5)
where F represents the cumulative distribution function (CDF) of the fore-
casted variable ( ˆXt0 +ω
c,i,j ), and H is an indicator function. The indicator function
equals 1 if the statement Xt0 +ω
c,i,j ↓ z is true; otherwise takes the value of 0
[44]. For deterministic forecasts, the CRPS reduces to the mean absolute error
(MAE) [ 42]. The xskillscore Python package is used to calculate the CRPS
metric. And we assume that the distribution of ensemble members follows a
Gaussian distributions, and the CRPS is computed based on the ensemble
mean and the ensemble variance. On the other hand, the SSR measures the
consistency between the spread of the ensemble and the RMSE of the EM.
The ensemble spread is deﬁned as:
Spread(c, ω)= 1
| D |
∑
t0 ↔D
√ 1
H → W
H∑
i=1
W∑
j=1
aivar( ˆXt0 +ω
c,i,j ) (6)
where var( ˆXt0 +ω
c,i,j ) denotes the variance within the ensemble dimension. A reli-
able ensemble is indicated by a SSR of one [ 45]. Lower values suggest an
underdispersive ensemble forecast, while higher values indicate overdispersion.

## Results
For evaluating FuXi’s performance, the study uses the 2018 data and selects
two daily initialization times (00:00 UTC and 12:00 UTC) to produce 6-hourly
forecasts for 15 days.

### Deterministic forecast metrics comparison
This subsection compares the forecast performance of FuXi, ECMWF HRES,
GrahpCast (the state-of-the-art ML-based weather forecast), ECMWF EM,
and FuXi EM on deterministic metrics. FuXi and GraphCast signiﬁcantly outperform
ECMWF HRES. FuXi and GraphCast have comparable performance within
forecasts of 7 days, beyond which FuXi shows superior performance, with the
lowest values of RMSE and the highest values of ACC across all the vari-
ables and forecast lead times. Moreover, FuXi’s superior performance becomes
increasingly signiﬁcant as lead times increase. Using an ACC value of 0.6 as
the threshold to measure a skillful weather forecast, we ﬁnd that FuXi extends
the skilful forecast lead time compared to ECMWF HRES, especially pushing
the lead time of Z500 and T 2M from 9.25 and 10 days to 10.5 and 14.5 days,
respectively.

Figure 3 shows the time series of the globally-averaged latitude-weighted
ACC and RMSE of ECMWF EM, FuXi, and FuXi EM, as well as the cor-
responding normalized di!erences in ACC and RMSE for 4 variables. The 4
variables include 2 upper-air variables ( Z500 and T 850) and 2 surface variables
(i.e., MSL and T 2M ). Many combinations of variables and pressure levels are
not included in the comparisons as they are unavailable from the ECMWF
server. The normalized di!erences in ACC and RMSE are computed using
ECMWF EM as the reference. FuXi superior performance to ECMWF EM in 0-9 day forecasts, with positive
values in the normalized ACC di!erence and negative values in normalized
RMSE di!erence. However, for forecasts beyond 9 days, FuXi shows slightly
poorer performance compared to ECMWF EM. Overall, FuXi shows compa-
rable performance to ECMWF EM in 15-day forecasts, with higher ACC and
lower RMSE than ECMWF EM on 67.92% and 53.75% of the 240 combi-
nations of variables, levels, and lead times in the testing set, which includes
2 surface variable and 2 upper-air variables over 15 days, with 4 steps each
day. The higher percentage of ACC could potentially be attributed to the fact
that the climatological mean used in the computation of the ACC is based on
ERA5 data, which serves as the ground truth for training FuXi. FuXi EM is
slightly inferior to the Fuxi deterministic forecast within short lead times for
all variables shown. However, it performs better after the lead time surpasses 3 days, which aligns with Pangu-Weather and FourCastNet.

Figure 4 illustrates the spatial distributions of the average RMSE of FuXi,
the RMSE di!erence between ECMWF HRES and FuXi, and the RMSE dif-
ference between ECMWF EM and FuXi for forecasts of Z500 and T 2M at lead
times of 5 days, 10 days, and 15 days, respectively. All forecasts in the test-
ing data from 2018 were averaged to produce the data. The RMSE di!erence
is represented by red, blue, and white patterns indicating whether ECMWF
HRES or ECMWF EM performs worse than, better than, or equally compared
to FuXi. Overall, all three forecasts have similar spatial error distributions,
with the RMSE di!erence values much lower than the RMSE values. The high-
est RMSE values appear at high latitudes, while relatively small values are
found in middle and low latitudes. The values of RMSE are higher over the
land than over the ocean. The RMSE di!erence between ECMWF HRES and
FuXi shows that FuXi outperforms ECMWF HRES in most grid points. In contrast, ECMWF EM shows comparable performance to FuXi in most areas.

### Ensemble forecast metrics comparison
Compared to deterministic forecasts, ensemble forecasts have several advan-
tages. They provide a more accurate EM than deterministic forecast in terms
of deterministic metrics and also represent forecast uncertainty through the
ensemble spread. This subsection focuses on comparing ensemble evaluation
metrics between the FuXi ensemble and the ECMWF ensemble. The CRPS values
for the FuXi ensemble are comparable to those of the ECMWF ensemble and
slightly smaller before 9 days. However, beyond 9 days, the FuXi ensemble
demonstrates inferior CRPS compared to the ECMWF ensemble. The SSR
values for the FuXi ensemble are signiﬁcantly higher than 1 for the 3 variables
such as Z500, T 850, and MSL in early lead times, indicating overdisper-
sion. These values then decrease dramatically with increasing lead times, and
becomes lower than 1, indicating an underdispersive ensemble. Meanwhile, the
SSR of the ECMWF ensemble are very close to 1, except for T 2M . Both the
FuXi ensemble and the ECMWF ensemble show underdispersion for T 2 as
their SSR values remain smaller than 1 throughout the 15-day forecast. While
the ensemble spread of the ECMWF ensemble grows as the forecast lead time
increases, the ensemble spread of the FuXi ensemble initially increases as the
lead time increases, then decreases after 9 days. One plausible explanation is
that the initial conditions are perturbed by the addition of Perlin noise, which
is random and independent of the background ﬂow. As a result, only a small
fraction of the perturbations remains after 9 days of model integration, causing
the ensemble spread to decrease.

## Conclusion and Future Work
It has been challenging for data-driven methods to compete with conventional
physics-based numerical weather prediction models in weather forecasting due
to the di”culty in reducing accumulation error. Recently, ML-based weather
forecasting systems have witnessed signiﬁcant breakthroughs, outperforming
ECMWF HRES in 10-day forecasts with a temporal resolution of 6 hours and
a spatial resolution of 0 .25→ [16, 17]. However, employing a single model proves
insu”cient to obtain optimal performance across various lead times. In order
to generate skillful weather forecasts for longer lead times, such as 15 days, we
ﬁrst develop a powerful base ML model architecture, FuXi model. The FuXi
model is based on the U-Transformer and has the capability to e”ciently learn
complex relationships from vast amounts of high-dimensional weather data.
Moreover, we propose a novel cascade ML model architecture for weather fore-
casting that utilizes three pre-trained FuXi models. Each model is ﬁne-tuned
for optimal forecast performance for one of the forecast time windows: 0-5
days, 5-10 days, and 10-15 days. These models are then cascaded to generate
comprehensive 15-day forecasts. By implementing the aforementioned method-
ologies, we created FuXi, an ML-based weather forecasting system that, for
the ﬁrst time, performs comparably to ECMWF EM in 15-day forecasts with
a temporal and spatial resolution of 6 hours and 0 .25→. Additionally, the FuXi
ensemble forecast exhibits promising potential, with a comparable CRPS to
ECMWF ensemble within 9 days for Z500, T 850, MSL , and T 2M .

In this study, we incorporate Perlin noise into the initial conditions to gen-
erate ensemble forecasts. The Perlin noise is random and independent of the
background ﬂow. Previous studies [ 46, 47] have shown that ﬂow-independent
initial perturbations decay over time during the model integration. Conse-
quently, to ensure an adequate ensemble spread in the medium range, we will
investigate ﬂow-dependent methods for initial condition perturbations in order
to maintain a reasonable spread throughout longer lead times for the FuXi
ensemble.

Furthermore, we plan to explore the potential of utilizing the cascade
ML model architecture for sub-seasonal forecasting. This will involve ﬁne-
tuning additional models for forecast lead times ranging from 14 to 28 days.
Sub-seasonal forecasting remains a challenge and is considered as a ”pre-
dictability desert” [ 48]. Unlike medium-range weather forecasting, which can
utilize deterministic methods, ensemble forecasts are necessary for sub-seasonal
forecasting. In addition, research has identiﬁed various processes in the atmo-
sphere, ocean, and land that contribute to sub-seasonal predictability, such as
the Madden-Julian Oscillation (MJO), soil moisture, snow cover, Stratosphere-
troposphere interaction, and ocean conditions [ 49]. Therefore, more research
is needed to develop an ML-based sub-seasonal forecasting system.

In addition, one limitation of current ML-based weather forecasting meth-
ods is that they are not yet completely end-to-end. They still rely on analysis
data generated by conventional NWP models for initial conditions. Thus, we
aim to develop a data-driven data assimilation method that uses observation
data to generate initial conditions for ML-based weather forecasting systems.
Looking to the future, we aim to build a truly end-to-end, systematically
unbiased, and computationally e”cient ML-based weather forecasting system.

## Data Availability Statement
We downloaded a subset of the ERA5 dataset from the o”cial website
of Copernicus Climate Data (CDS) at https://cds.climate.copernicus.eu/.
The ECMWF HRES forecasts are available at https://apps.ecmwf.
int/archive-catalogue/?type=fc&class=od&stream=oper&expver=1 and
ECMWF EM are available at https://apps.ecmwf.int/archive-catalogue/
?type=em&class=od&stream=enfo&expver=1. The preprocessed sam-
ple data used for running FuXi models in this work are available
in a Google Drive folder ( https://drive.google.com/drive/folders/
1NhrcpkWS6MHzEs3i lsIaZsADjBrICYV).

## Code Availability Statement
We used the code base of Swin transformer V2 as the backbone archi-
tecture, available at https://github.com/microsoft/Swin-Transformer. The
source code used for training and running FuXi models in this work is
available in a Google Drive folder ( https://drive.google.com/drive/folders/
1NhrcpkWS6MHzEs3i lsIaZsADjBrICYV). The aforementioned Google
Drive folder contains the FuXi model, code, and sample input data, which
can be accessed by individuals with the provided link. As the FuXi model
and code are essential resources for this study, we have implemented pass-
word protection for the Google Drive folder link through a Google Form.
To obtain the link to the Google Drive folder from the Zenodo link,
users are required to complete the designated Google Form.