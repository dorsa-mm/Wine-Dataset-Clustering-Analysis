# Wine Dataset Clustering

## Overview
This project applies **unsupervised machine learning techniques** to cluster wines based on their chemical properties. The dataset contains **various chemical compositions** of different types of wines, and the goal is to identify meaningful groupings using **K-Means and Hierarchical Clustering**.

## Objectives
- Perform **data preprocessing** by standardizing the features.
- Implement **K-Means clustering** and **Hierarchical clustering**.
- Evaluate clustering performance using **Silhouette Scores**.
- Visualize the results using **Principal Component Analysis (PCA)** and a **Dendrogram**.

## Dataset
The dataset contains **wine chemical compositions** and is structured as follows:
- **Class**: The actual wine category (removed for unsupervised learning).
- **Alcohol, Malic Acid, Ash, etc.**: Chemical properties of wines.

The dataset is stored in a CSV file and is loaded into a Pandas DataFrame.

## Methods Used
### 1. **Data Preprocessing**
- The dataset is loaded into a Pandas DataFrame.
- The `Class` column is dropped as clustering is **unsupervised**.
- Features are standardized using **StandardScaler** to normalize different scales.

### 2. **K-Means Clustering**
- K-Means is applied with **3 clusters** (as the dataset originally contains 3 wine classes).
- Cluster assignments are made using `fit_predict`.
- **Silhouette Score** is computed to evaluate clustering effectiveness.

### 3. **Hierarchical Clustering**
- Agglomerative Clustering with **Ward’s linkage** is used.
- The Euclidean distance metric is applied.
- The **Silhouette Score** is calculated to measure the quality of clustering.

### 4. **Visualization**
- **PCA (Principal Component Analysis)** reduces feature dimensions to **2D** for plotting K-Means clusters.
- A **dendrogram** is generated for hierarchical clustering to show cluster formation.

## Results
- **K-Means Silhouette Score**: Measures clustering compactness and separation.
- **Hierarchical Silhouette Score**: Evaluates cluster distinctiveness.
- **Cluster Centers**: Displays centroids of K-Means clusters.

## Installation & Setup
### **Prerequisites**
Ensure you have Python installed along with the following libraries:
```sh
pip install pandas scikit-learn matplotlib scipy
