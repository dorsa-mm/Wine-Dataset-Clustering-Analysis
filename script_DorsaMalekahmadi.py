import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Using PCA to reduce dimensions
from scipy.cluster.hierarchy import dendrogram, linkage  # For dendrogram

# Load the wine dataset
wine_data_path = "C:\\Users\\dorsa\\Desktop\\wine\\wine.data"
columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

# Load dataset into a DataFrame
df = pd.read_csv(wine_data_path, header=None, names=columns)

# Drop the 'Class' column since clustering is unsupervised
features = df.drop("Class", axis=1)

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(features_scaled)
kmeans_silhouette = silhouette_score(features_scaled, kmeans_labels)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
hierarchical_labels = hierarchical.fit_predict(features_scaled)
hierarchical_silhouette = silhouette_score(features_scaled, hierarchical_labels)

# Generate summary of results
results_summary = {
    "K-Means Silhouette Score": kmeans_silhouette,
    "Hierarchical Silhouette Score": hierarchical_silhouette,
}

# Visualizing the K-Means clustering result in a 2D space 
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Scatter plot of K-Means clusters
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_labels, cmap="viridis", s=50) #: Viridis is a colormap to assign colors to clusters
plt.title("K-Means Clustering (2D Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.grid()
plt.show()

# Dendrogram for hierarchical clustering (Ward's method minimizes SSE when merging clusters)
linkage_matrix = linkage(features_scaled, method="ward")

plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.grid()
plt.show()

# Print Silhouette Scores
print("K-Means Silhouette Score:", kmeans_silhouette)
print("Hierarchical Silhouette Score:", hierarchical_silhouette)

# Print K-Means Cluster Centers
print("\nK-Means Cluster Centers:")
print(kmeans.cluster_centers_)
