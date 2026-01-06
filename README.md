# ML-Project-312881
Machine Learning Project 2025/2026 - Court Dynamics

### Group Members:

- Federico Sanna: 312881

- Arianna Di Lecce: 319181

- Filippo Parissi: 309151

## Section 1: Introduction

This project explores the design and implementation of a machine learning–based system aimed at addressing a specific real-world or analytical problem through data-driven methods. The primary objective of the project is to develop, evaluate, and analyze a model (or set of models) that can effectively learn patterns from data and produce meaningful insights.

Our work focuses on investigating how different design choices—such as feature representation, model architecture, and training strategies—impact overall system performance. By structuring the project as an end-to-end pipeline, we aim to provide a clear and reproducible workflow that spans data preprocessing, model training, evaluation, and result interpretation. Ultimately, this project serves both as a practical application of machine learning concepts and as an experimental framework for understanding the strengths and limitations of the proposed approach.

## Section 2: Methods

### 2.1 Data Preprocessing

Data preprocessing represents a critical step in the proposed pipeline, as the raw dataset contains missing values, heterogeneous data types, skewed distributions, and features with very different numerical scales. The goal of this step is to clean, standardize, and transform the data in order to make it suitable for downstream modeling while preserving as much information as possible.

The preprocessing pipeline includes:
1) analysis of missing values
2) imputation of numerical features
3) inspection of feature distributions before and after imputation
4) verification of missingness removal
5) analysis of data types
6) outlier and scale inspection
7) feature scaling.

#### Data Type Inspection

<img width="976" height="476" alt="3" src="https://github.com/user-attachments/assets/3cf96430-2c5e-4d33-8806-1ce7fb08b7bd" />

An inspection of feature data types reveals that the majority of features are numerical (float64), with a smaller subset of categorical (object) and integer features. 

⸻

#### Missing Values Analysis (Before Preprocessing)

<img width="1175" height="576" alt="1" src="https://github.com/user-attachments/assets/748db563-dce7-4102-83f9-d25e5a3968ab" />

Figure description:
Missing values rate per column before preprocessing.

Before applying any transformation, we analyze the extent and distribution of missing values across all features. The dataset exhibits a substantial amount of missing data, particularly among team-level and advanced statistical features. This analysis motivates the need for a systematic imputation strategy rather than simple row-wise deletion, which would result in significant information loss.

⸻

#### Missingness Structure (Before Preprocessing)

<img width="1076" height="368" alt="2" src="https://github.com/user-attachments/assets/22d47973-fa35-40c7-9696-64f4b51518d4" />

Figure description:
Missingness matrix before preprocessing.

⸻

#### Numerical Imputation Strategy - Before Median Imputation

<img width="1375" height="776" alt="4" src="https://github.com/user-attachments/assets/e51decfc-467c-4a8c-98d4-723bb07a0dee" />

Figure description:
Distribution of key numerical features before median imputation.

Given the skewed distributions and presence of outliers, median imputation is applied to numerical features. The median is preferred over the mean as it is more robust to extreme values and better preserves the central tendency of non-Gaussian distributions.

This choice ensures that imputation minimally distorts the original feature distributions while allowing all samples to be retained.

#### Numerical Imputation Strategy - After Median Imputation

<img width="1375" height="776" alt="5" src="https://github.com/user-attachments/assets/aaedf522-f18d-4882-bbac-58a4fa8f64aa" />

Figure description:
Distribution of key numerical features after median imputation.

After median imputation, the overall shape of the feature distributions is preserved. Peaks introduced by imputation correspond to the median values, but no artificial smoothing or distortion is observed. This confirms that the imputation strategy maintains the statistical characteristics of the data.

⸻

#### Numerical Feature Scaling – Before Scaling

<img width="1376" height="476" alt="7" src="https://github.com/user-attachments/assets/e592e9bd-f1f4-4559-a4f8-6438243d7cb5" />

Figure description:
Boxplots of numerical features before scaling.

Before scaling, numerical features exhibit very different ranges and variances (e.g., minutes vs. blocks). Such discrepancies can negatively impact distance-based and gradient-based models. Boxplots also highlight the presence of extreme values and outliers.

⸻

#### Numerical Feature Scaling – After Scaling

<img width="1376" height="476" alt="8" src="https://github.com/user-attachments/assets/8054f2c6-43f6-4646-b95d-e332cb3548ed" />

Figure description:
Boxplots of numerical features after scaling.

After applying feature scaling, numerical variables are brought to a comparable scale while preserving relative differences and outlier structure. This step improves numerical stability and ensures that no single feature dominates the learning process due to scale alone.

⸻

#### Missingness Structure (After Preprocessing)

<img width="1076" height="368" alt="6" src="https://github.com/user-attachments/assets/4c8964e1-0950-4500-9538-ec64b8ea0dc7" />

Figure description:
Missingness matrix after preprocessing.

The post-imputation missingness matrix shows that all missing values have been successfully addressed. The dataset is now complete and suitable for machine learning models that do not natively handle missing data.

### 2.2 Exploratory Data Analysis (EDA)

The purpose of the Exploratory Data Analysis (EDA) step is to gain a deeper understanding of the dataset after preprocessing, focusing on feature distributions, temporal trends, relationships between variables, and structural dependencies. This analysis guides subsequent modeling decisions by revealing statistical properties, correlations, and potential sources of bias or redundancy.

⸻

#### League-Level Trends Over Time

<img width="1176" height="576" alt="12" src="https://github.com/user-attachments/assets/61084b2e-0ce5-4900-ac7d-e7a3cc9a900e" />

Figure description:
League-level trends of average player performance over time for points, rebounds, and assists.

To capture temporal dynamics, we analyze league-level trends by aggregating player statistics across seasons. This graph shows league-wide average trends in points, rebounds, and assists per player over time. All three stats rise from below average in earlier seasons, peak around the mid-period, and then gradually decline toward the league average. A sharp dip appears in the later seasons, suggesting a league-wide disruption, followed by a partial recovery.

⸻

#### Correlation Analysis of Performance Metrics

<img width="662" height="576" alt="13" src="https://github.com/user-attachments/assets/7f029bce-b0e4-41d6-8168-0faefd7a98c4" />

Figure description:
Correlation matrix of key performance metrics.

Next, we explore linear relationships between numerical features using a correlation matrix. Strong positive correlations are observed between minutes played and most performance metrics, particularly points and rebounds, indicating that playing time is a major driver of raw statistical output. In contrast, some metrics (e.g., assists and blocks) exhibit weaker or more nuanced relationships, suggesting that they capture distinct aspects of player behavior. This analysis helps identify potential multicollinearity issues and informs feature selection.

⸻

#### Relationship Between Playing Time and Scoring Output

<img width="976" height="476" alt="14" src="https://github.com/user-attachments/assets/d178b2a6-9dae-4893-8a11-fbad626cbee8" />

Figure description:
Scatter plot of points scored versus minutes played.

Finally, we explicitly analyze the relationship between playing time and scoring output. The scatter plot shows a clear positive relationship, but also highlights substantial variability among players with similar minutes played. This indicates that while playing time is a necessary condition for scoring, it is not sufficient to fully explain scoring performance. This observation motivates the inclusion of additional contextual and skill-related features in the predictive models.

### Section 3 – Experimental Design

This section describes the experimental setup used to validate the target contributions of the project. The experiments focus on identifying meaningful player groupings using unsupervised learning techniques and evaluating their interpretability and stability under different clustering approaches.

⸻

#### Experiment 1: K-Means Clustering on Player Performance Profiles

- Main purpose:
The goal of this experiment is to identify distinct player archetypes based on standardized performance metrics, using K-Means clustering as a baseline unsupervised method.

- Baseline(s):
K-Means is used as the primary baseline due to its simplicity, scalability, and widespread use in clustering continuous numerical data.

- Evaluation metrics:
Evaluation is qualitative and structural, based on:
  - Cluster separation in reduced-dimensional space (PCA projection)
  - Cohesion around centroids
  - Interpretability of resulting clusters in terms of basketball performance

<img width="1176" height="676" alt="15" src="https://github.com/user-attachments/assets/c99b868b-4e0b-4480-81aa-0a41dcac1f63" />

Figure description:
K-Means clustering visualized in PCA space with cluster centroids.

The PCA projection allows visualization of high-dimensional player data in two dimensions. The presence of relatively compact and well-separated clusters suggests that K-Means captures meaningful structure in the data, with centroids representing distinct player profiles.

⸻

#### Experiment 2: Hierarchical Clustering as a Structural Baseline

- Main purpose:
This experiment explores hierarchical relationships between players to assess whether natural multi-level groupings exist in the data.

- Baseline(s):
Agglomerative hierarchical clustering with Ward linkage is used as a structural baseline, offering a different perspective compared to partition-based methods.

- Evaluation metrics:
  - Dendrogram structure and merge distances
  - Visual identification of meaningful cluster splits
  - Comparison with flat clustering results

<img width="1176" height="576" alt="16" src="https://github.com/user-attachments/assets/ee2a7ac9-5c9c-44d4-868d-3e7d2677b291" />

Figure description:
Agglomerative clustering dendrogram (Ward linkage, subsampled data).

The dendrogram reveals a hierarchical organization of player profiles, highlighting how clusters progressively merge. This provides insight into cluster granularity and supports the choice of a fixed number of clusters for K-Means.

⸻

#### Experiment 3: Density-Based Clustering with DBSCAN

- Main purpose:
The objective of this experiment is to identify dense regions of similar player profiles while explicitly detecting outliers and atypical players.

- Baseline(s):
DBSCAN is compared against K-Means to evaluate the impact of density-based assumptions and noise handling.

- Evaluation metrics:
  - Ability to identify noise points
  - Shape flexibility of detected clusters
  - Qualitative comparison with centroid-based clustering

<img width="1176" height="676" alt="17" src="https://github.com/user-attachments/assets/648346c5-5dc9-4fde-b960-c54a0e664efe" />

Figure description:
DBSCAN clustering results visualized in PCA space, including noise points.

The results show that DBSCAN identifies a core dense region while labeling a significant portion of players as noise. This highlights the heterogeneity of player profiles and the limitations of density-based methods in highly skewed, continuous performance data.

⸻

#### Experiment 4: DBSCAN Hyperparameter Intuition via k-Distance Plot

- Main purpose:
This experiment supports the DBSCAN setup by providing intuition for the selection of the eps parameter.

- Baseline(s):
No alternative method is used; this analysis serves as a diagnostic tool for DBSCAN configuration.

- Evaluation metrics:
  - Visual identification of the “knee” point in the k-distance curve
  - Stability of clustering results around selected eps

<img width="976" height="376" alt="18" src="https://github.com/user-attachments/assets/d3792f02-7e83-463f-bd08-0033668143e4" />

Figure description:
k-distance plot (k = 5) used to estimate the DBSCAN epsilon parameter.

The sharp change in slope indicates a natural threshold separating dense regions from sparse areas. This visualization guides the selection of a reasonable eps value for DBSCAN.

⸻

#### Experiment 5: Cluster Validation on Held-Out Test Data

- Main purpose:
The goal of this experiment is to assess whether the learned clusters generalize to unseen data and maintain interpretability.

- Baseline(s):
K-Means clustering learned on the training set is applied to the test set without re-fitting.

- Evaluation metrics:
  - Consistency of cluster structure
  - Visual coherence in feature space
  - Stability of cluster assignments

<img width="976" height="576" alt="19" src="https://github.com/user-attachments/assets/c6631791-8827-442d-9691-f06873ff5c47" />

Figure description:
Final K-Means clusters on the test set: Minutes vs Points.

The test-set visualization confirms that clusters remain well-structured and interpretable, indicating that the clustering solution generalizes beyond the training data.
