# ML-Project-312881
Machine Learning Project 2025/2026 - Court Dynamics

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

