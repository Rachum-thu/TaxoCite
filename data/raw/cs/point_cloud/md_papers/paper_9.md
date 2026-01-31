# Semantic Segmentation on Swiss3DCities: A Benchmark Study on Aerial Photogrammetric 3D Pointcloud Dataset

## Abstract
We introduce a new outdoor urban 3D pointcloud dataset, covering a total area of 2.7 km2, sampled from three Swiss cities with different characteristics. The dataset is manually annotated for semantic segmentation with per-point labels, and is built using photogrammetry from images acquired by multirotors equipped with high-resolution cameras. In contrast to datasets acquired with ground LiDAR sensors, the resulting point clouds are uniformly dense and complete, and are useful to disparate applications, including autonomous driving, gaming and smart city planning. As a benchmark, we report quantitative results of PointNet++, an established point-based deep 3D semantic segmentation model; on this model, we additionally study the impact of using different cities for model generalization.

## 1 Introduction
Many recent achievements of deep learning depend on the availability of very large labeled training datasets [29], [10], such as ImageNet [8] for image classification and MS COCO [18] for image segmentation. In this work, we propose a new dataset of dense urban 3D pointclouds, spanning 2.7 km2, acquired using photogrammetry from three different cities in Switzerland (Zurich, Zug and Davos). The entire dataset is manually annotated with dense labels, which associate each point to one of five categories: terrain, construction, vegetation, vehicle, and urban asset.

The main goal of the dataset is to train and validate semantic segmentation algorithms for urban environments. Semantic segmentation consists in partitioning the data into multiple sets of points, such that each set represents only objects of a given type. The problem is relevant for many real-world applications, such as autonomous or assisted driving, automated content generation for games [15], augmented and virtual reality applications, and city planning [38].

Most existing datasets [27], [3], [32] for outdoor 3D semantic segmentation are motivated by real-time autonomous driving applications, and are therefore acquired at low resolution by street-level Light Detection and Ranging (LiDAR) sensors; this yields incomplete point clouds (for example, areas far from roads, such as roofs, are either not acquired or acquired with very low resolution) which are unsuitable for applications such as city planning, urban augmented or virtual reality (AR/VR), or gaming. In contrast, we acquire high-resolution photographs from unmanned aerial vehicles (UAV) flying a grid pattern over the area of interest, then reconstruct the 3D shape using photogrammetry; this allows us to densely acquire most outdoor surfaces. Similar approaches have been previously adopted for several applications, including automatic urban area mapping [20], damage detection [19], and cultural heritage site mapping for digital preservation [23]. Compared to 3D models built by satellite-borne cameras, this approach yields models with higher-resolution geometry and texture.

High resolution data yields more accurate models, but also aids the segmentation task because it contains more information to discriminate between different classes; currently, state-of-the-art models for 3D semantic segmentation rely on deep learning [11], [39], [4] and represent input data as voxels [40], points [24] or meshes [17]; other approaches render multiple views of the 3D scene and then rely on 2D semantic segmentation models [30], [31], [5], which can be trained on more abundant 2D labeled semantic segmentation datasets.

To show the potential of our dataset for training and evaluating segmentation algorithms, we consider the well-established PointNet++ model [24], [25] and report its performance when using different splits for training and evaluation. In particular, the performance of machine learning models depends not only on the size of the training dataset, but also on how representative it is of the evaluation data: often, models trained on large amounts of data from a given environment fail to generalize to a different target environment. Because our dataset contains data from three cities with different characteristics, it can be used to explore this fundamental aspect.

The rest of the paper is organized into five sections. We first describe related commonly-used datasets for 3D semantic segmentation in Section 2. Then, in Section 3 we present our main contribution: a new pointwise labeled multi-city dataset for semantic segmentation of outdoor 3D point clouds, which we release to the research community in three versions with different point densities; we characterize the dataset and describe data acquisition, processing and manual labeling pipelines. In Section 4 we describe the applied deep learning model to demonstrate semantic segmentation task on our dataset. We discuss quantitative results in Section 5, where we also explore the model’s generalization ability across different cities (secondary contribution). Section 6 concludes the paper.

## 2 Related Work
This section summarizes relevant pointcloud datasets with semantic segmentation labels (see Table 1). One fundamental difference among the datasets is their acquisition modality, i.e. LiDAR or photogrammetry.

### 2.1 LiDAR Datasets
A lot of recent research efforts are related to autonomous driving applications: in particular, recognizing and segmenting roads and relevant urban elements from images or 3D point clouds acquired by the car itself. In this context, laser scanning systems, e.g. Velodyne HDL-64E [34], are commonly used to acquire high-accuracy LiDAR pointcloud sequences from a car’s point of view. Paris-Lille[27], Semantic KITTI[3], and Toronto3D dataset[32] are among such large-scale datasets with pointwise semantic labels.

Due to the low-lying viewpoint and focus on driving-related segmentation tasks, these mobile LiDAR datasets show incomplete point clouds: e.g. the upper floors or roofs of the buildings are usually not captured. Even though these datasets serve their main scope very well, they are not suitable for other applications, such as urban planning.

Semantic3D dataset[12] is a large-scale pointcloud dataset with per-point semantic labels. This dataset is acquired via a static terrestrial laser scanning system in the north-east of Switzerland. Several points to note about this dataset are gaps due the occlusions (also known as LiDAR shadows), moving object artifacts, and varying point density based on the distance of the laser system to each surface or object in the scene.

As captured from air (either from a UAV or a helicopter), aerial LiDAR datasets such as ISPRS airborne LiDAR pointcloud dataset [21], DublinCity dataset [41] and DALES dataset [33] are also relevant in our context. One important difference of these datasets with respect to ours is that, due to the narrow divergence of laser beams, they can sometimes capture ground samples even when covered by vegetation. Compared to the DublinCity aerial LiDAR dataset[41], ours covers a moderately larger area and, in the medium-density version, has a similar point density. This shows the relevance of our contribution with respect to existing aerial LiDAR datasets.

Photogrammetric pointclouds Sun3D [36] and Stanford Large-Scale Indoor Spaces 3D (S3DIS) [1] are commonly used pointcloud datasets acquired using Structure-from-motion 3D reconstruction techniques. These datasets are focused on indoor scenes, and present interesting challenges for computer vision research, such as the presence of clutter, and relevant context around different objects, that can play a role in scene understanding. Due to their limited extent, the capturing process is much less challenging than in large-scale outdoor contexts, which also need to account for variability of weather, illumination conditions, and scales of represented objects.

The Pix4D dataset [2] comprises of aerial photogrammetric pointclouds from three outdoor scenes with different distributions of urban surfaces or objects. The authors emphasize the importance of color features apart from geometric features to classify these pointclouds into 6 semantic classes. This dataset is relatively small-scale, since it comprises of only three scenes with a total of 18.2 million points.

The SenSatUrban dataset [16] is also reconstructed via photogrammetry from aerial photographs. The photographs are taken with a UAV that follows a double-grid flight path and covers a 6 km2 area in three cities in UK (Birmingham, Cambridge, and York). Pointwise semantic labels in 13 categories are available for these pointclouds. As an urban-focused aerial photogrammetric pointcloud, the SenSatUrban dataset is the most relevant with respect to our contribution. SenSatUrban covers an approximately twice larger area than ours, and uses 13 categories instead of five; its point density is higher than our medium-density version, but lower than our high-density version.

## 3 Dataset Description
In this section, we describe the process used to produce our large scale aerial photogrammetry dataset, covering both acquisition of source photographs and processing to obtain 3D point clouds. We conclude the section by detailing the data characteristics.

### 3.1 Data Acquisition
The image data is acquired via a high-resolution camera array (nadir and oblique cameras) mounted on a multirotor drone.

To capture the image data, the drone is configured to trigger the cameras simultaneously at regular intervals. The drone follows a double grid flight path [28].

Each flight acquires one or more tiles. A single tile corresponds to 412 m × 412 m horizontal area approximately (around 17 hectares). The Ground Sampling Distance (GSD), i.e. the inter-pixel distance measured on the ground, is planned as 1.25 cm and ultimately measured as 1.28 cm.

### 3.2 Data Processing
After aerial image acquisition, we follow a classic photogrammetry workflow to reconstruct textured 3D models, based on RealityCapture [26], [14], a commercial photogrammetry software.

We first estimate the global camera poses of the captured images and a georeferenced sparse point cloud of the scene using a standard Structure-from-Motion (SfM) process (referred as "alignment" in RealityCapture). Georeferencing is achieved using Ground Control Points (GCPs) and RealityCapture’s GCP annotation tool. Taking into account the drone-based acquisition described above, the GCP annotation, and the further processing of the data, we measure a total georeferencing root mean square error (RMSE) of 5.45 cm horizontally and 11.60 cm vertically. This implies that the data is scaled to real world units, meters in our case. Note that georeferencing is not a priority for semantic segmentation task apart from scaling the data to the real world units. Therefore, we provide all point coordinates as scaled and, for convenience, as zero-centered per tile in our dataset.

Once the data is aligned and georeferenced, we reconstruct a dense mesh constrained to the geographic region of the tile only. At that point, the raw mesh obtained can contain up to half a billion polygons for a single tile. In order to get a mesh with a more manageable size, we simplify it to a maximum number of 30 million polygons, i.e. approximately 15 million vertices, with RealityCapture’s adaptive simplification process and texture it using the captured drone images. The output point cloud used for segmentation is composed of the vertices of such a mesh; the RGB color of each point is sampled from the mesh texture.

### 3.3 Manual Segmentation
Our pointclouds are segmented manually into the five semantic categories as described below (terrain, construction, vegetation, vehicles and urban assets).

3D artists complete this task using off-the-shelf 3D modeling software (such as Blender [7]); to make the process manageable, they work on each tile individually, and operate on a 1-million polygon mesh further simplified from the initial mesh. It takes between six to twelve hours for a 3D artist to manually segment each tile.

Labels are then transferred from the simplified mesh to the output point cloud. The label of each point in the output point cloud is assigned by finding the nearest neighbor in the segmented mesh. We used an adaptive distance threshold to avoid matching outlier points. We found that this method gives satisfying results for the final segmentation of the point cloud while keeping the amount of manual work needed at a manageable level.

### 3.4 Dataset details
The dataset represents sixteen tiles acquired from three cities in Switzerland: six tiles from Zurich, five tiles from Zug and five tiles from Davos.

For each tile, the dataset contains pointclouds at three resolutions, i.e. approximately 500 K, 15 M, and 225 M points per tile. Both 500K and 15M point density pointclouds have x,y,z, and RGB color features. For the highest density, we have only x,y,z coordinates. In the rest of this paper, we only consider the 15M point density.

Classes and class distribution As our dataset is focused on urban areas in Switzerland, it comprises of a large amount of terrain, building, and medium or high vegetation. Even though many objects of other categories (such as vehicles or urban assets) are present in our dataset, they amount to a relatively small portion of the points because each object is relatively small. For that reason, we divide our semantic labeling to only five main categories: 1) terrain (including natural terrain, e.g. grass or soil, impervious terrain, e.g. road or sidewalk, and water areas, e.g. river or lake); 2) building; 3) urban asset (including traffic light, pole, crane, public transportation stop, trash bin, etc.); 4) vegetation (tree or bush); and 5) vehicle (car, bike, scooter, etc.). The total number of points per category can be found in the dataset.

## 4 Semantic Segmentation
To exemplify the usage of the proposed dataset for training and evaluating semantic segmentation models, and to provide baseline performance metrics, we report experiments using PointNet++ [25], a well-established point cloud segmentation approach.

### 4.1 PointNet++
PointNet++ [25] is a deep learning model built upon the PointNet [24] model. In the PointNet++ architecture, PointNet module is used as a local feature encoder and applied in a nested fashion to learn hierarchical features. Moreover, PointNet++ uses farthest point sampling to cover more representative points during sampling.

We adopt an existing implementation [35] of PointNet++ developed using PyTorch [22], PyTorch-Lightning [9] and Hydra [37]. For a given instance, the input of the model is a N × 6 matrix, each row containing the x, y, z coordinates and R, G, B color of one of N input points. The output of the model is a matrix of N × K prediction probabilities, where K = 5 is the number of classes. Because the model is designed to handle input point clouds up to a few thousand points (N = 8192 in our reference implementation), it cannot be directly applied to our large outdoor datasets; therefore, we implemented the following data pipeline. First, we partition the input data into columns with a base of 10 m × 10 m and infinite height. During training and validation, each instance is generated by picking a column, then randomly sampling (with replacement) N points from the column. A training epoch is obtained by generating one instance per column. For every training epoch, the instances are sampled again; this yields a form of data augmentation since for each column a different subset of points is sampled in every epoch.

Once a model is trained, in order to segment a testing tile, we apply the model to every column separately, then merge the segmentation results. To segment a column, we randomly divide the points in the column in subsets, each containing exactly N points; for the last subset, in case less than N points are remaining, additional points are sampled from the other subsets. Each subset defines an instance an is segmented independently using the trained model; the results are then combined.

The model is trained by minimizing the cross-entropy loss; to deal with heavy class imbalance, following in similar works [16], the loss is weighted differently for each class, according to inverse-square-root frequency. A training batch is composed by 64 instances and we train for 200 epochs; we do not use early stopping but snapshot the model which yields the minimum loss on the validation set (which is defined on tiles different than training and testing tiles). Other hyper-parameters are set as in [35]. The experiments are run on a NVIDIA RTX 2080Ti GPU. The longest train and test sessions are completed in less than 6 hours.

### 4.2 Experimental Setup
Our experimental setup focuses on the following research questions, that are more related to the characteristics of the data than to the capabilities of the specific model.

- Which categories are more challenging to segment?
- How does the model generalize across cities?
- How much can additional data help even if it is from a different city?
- Which training strategy is better for pointcloud data: an ensemble of per-city models or a single model trained on all data from multiple cities?

To answer the questions above, we train four models: three on data sampled from a single city (named single-city models in the following); one on data from all three cities. Then, we apply each model on three testing sets (disjoint from the training and validation sets), one per city.

Data Splits In particular, we consider five tiles for each of the three cities. Each tile covers approximately 0.17 km2, which yields 0.855 km2 and 70 million points per city. For each city, the five tiles are partitioned in three tiles for training, one tile for validation and one tile for testing. Single-city models are therefore trained on three tiles and validated on one tile. The model trained on all cities is trained on nine tiles and validated on three tiles. Each of the four models is tested on three tiles (one per city), on which we separately compute performance metrics.

Evaluation Metrics For a given testing tile, a model under test will produce five class probabilities (which sum to 1) for each point. The point is then assigned to the class that has the largest probability. From these data, we compute the following commonly-used metrics [1], [12], [3] to quantify segmentation performance.

- Overall Accuracy is the fraction of the points for which the predicted class coincides with the ground truth class (also known as micro-averaged accuracy).
- Weighted Accuracy is the macro-averaged accuracy that is multiplied with a per-class factor. For a given class c, the factor is computed as the proportion of the number of class samples Nc over the number of samples in the whole dataset N, i.e. Nc/N.
- Per-class F1 score is the harmonic mean between per-class precision and recall. An F1 score of 1.0 indicates an ideal classifier.
- Per-class Intersection over Union score (IoU): For a given class c, the IoU score is computed as the ratio between: the number of points that have been classified as class c AND are indeed of class c (intersection); and the number of points have been classified as class c OR are indeed of class c (union). An IoU score of 1.0 indicates an ideal classifier.

For all per-class metrics, we also report average values among all the classes and the weighted averages. For consistency, we report all the metrics as percentage values (the ratios between 0 and 1 are scaled linearly between 0 and 100).

## 5 Results and Discussion

### 5.1 Overall Performance Metrics
On the three testing tiles, the model trained on all cities yields an overall accuracy of 82.8%, weighted accuracy of 87.6%, average F1 of 56.0%, and average IoU score of 45.3%.

### 5.2 Per-Category Performance
We observe that “urban asset” and “vehicle” classes are harder to segment compared to other classes; this is expected due to their small size, and widely variable characteristics in terms of shape and color. IoU metrics are particularly penalized, due to the small size of each object.

As illustrated in the confusion matrix, one can observe the following confusion cases among categories: 1) urban asset and other categories (especially construction), 2) vegetation and terrain, and 3) vehicle and construction categories. An additional urban test tile from Zurich exhibits similar trends to the overall confusion matrix (average of the three test tiles). However, one can notice the increase in the confusion trends and additional confusion between terrain and construction categories. This pointcloud exhibits particular urban characteristics such as a bridge, entrance to an underground parking lot, a botanical garden on a hill, and glass or plants/moss/soil covered rooftops. We consider that the lower amount of terrain in this urban setting also makes the confusion noticeable.

We hypothesize that a data pipeline that emphasizes the relative height information and favors the small categories in a stronger fashion than our current setting (e.g. a cube-based sampling rather than column-based sampling) and a stronger model than PointNet++ might help decreasing these confusion cases. As our goal is to report a baseline model on our novel dataset, we keep these model explorations for future work. For simplicity, we report and discuss only on the three testing tiles (one per city) further in this section.

### 5.3 Model Generalization across Cities
We analyze the model generalization in a cross-city experiment setting.

The performance of the M1 and M2 models, which were trained on rural or industrial areas in Davos and Zug, decreases when they are tested on the urban Zurich test tile. Similarly, the Zurich model (M3), which is trained with the urban pointclouds, i.e. high-rise large buildings, performs worse on the rural Davos test tile than the other test tiles. This trend is observed further in the ensemble results. This point emphasizes the importance of area characteristics while learning semantics.

Comparing models trained on data from the same city with models trained on data from a different city, we observe that the models trained on data from the same city have significantly better performance (average weighted accuracy 84.9%) than the models trained on data from a different city (average weighted accuracy 82.3%), despite the fact that the amount of training data is the same in different cities, and that areas used for training are always disjoint from areas used for testing.

### 5.4 Impact of Data Scale
Comparing models, we observe that the performance of the model M4 trained on data from all three cities (average weighted accuracy 87.6%) is better than the performance of the model trained just on data from the same city (average weighted accuracy 84.9%). This quantifies the impact of tripling the amount of training data, even though the additional data comes from two different cities.

### 5.5 Model Ensembling
We consider an alternative approach to training a single model on three cities; instead, we consider the three single-city models (M1, M2, and M3), apply each model independently to each test tile, then average their predictions; in particular, for each given point in the test tile, we obtain three class probability vectors as outputs of each of the three models; we compute the element-by-element average of the three vectors, which yields a single class probability vector whose 5 elements also sum to 1. This approach is known as model ensembling [13], [6] and is used frequently in machine learning.

The ensemble of the three single-city models outperforms the single model trained on the three cities. The ensembling approach is appealing, since training each model on a single-city dataset is simple and flexible: by averaging their results, we minimize the consequences of overfitting and more generally counteract model variance; on the other hand, the computational cost for inference is tripled, as three models have to be evaluated for each input.

Model ensembling experiments also allow us to quantify the performance gains from acquiring additional training data; in particular, by adding a model trained on a different city, we observe benefits; adding a third model to the ensemble shows decreasing returns, even if it is trained on the same city used for evaluation.

## 6 Conclusion
This paper introduces a novel urban pointcloud dataset with pointwise semantic groundtruth. The dataset is constructed via photogrammetry on UAV-acquired high-resolution images of three Swiss cities. The dataset reports three pointcloud densities: a simplified sparse pointcloud with RGB colors and semantic labels, a regular density pointcloud with RGB colors and semantic labels, and a dense pointcloud with only x,y,z coordinates (with potential applications e.g. in robotics for ground traversibility mapping).

The paper describes the acquisition and processing of the dataset, then illustrates several experiments on a semantic segmentation task with a prominent point-based deep learning benchmark model (PointNet++ [25]). These experiments highlight: 1) the importance of the amount of training data; 2) the advantage of using training data from the same city on which the model is evaluated; 3) the viability of simple model ensembling approaches.

As future work, we plan to compare additional recent deep-learning models for the semantic segmentation task on this dataset. Moreover, we plan to study the effects of semi-supervised and self-supervised learning methods on unstructured pointclouds.

As we make this dataset available to the research community, we hope that it will be useful for further analysis of model generalization, domain-gap studies with respect to LiDAR datasets, and various robotics applications such as traversibility, and ultimately advance the state of the art in the field.