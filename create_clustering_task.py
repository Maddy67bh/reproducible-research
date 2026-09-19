import os, json, subprocess, sys
import numpy as np
import pandas as pd

# Install required packages
packages = [
    "pandas","numpy","scikit-learn","matplotlib",
    "seaborn","plotly","jupyter","nbformat","nbclient"
]
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q"
] + packages)

os.makedirs("notebooks", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# ============================================================
# CREATE SYNTHETIC DATASET
# ============================================================
np.random.seed(42)

from sklearn.datasets import make_blobs

X, true_labels = make_blobs(
    n_samples=600,
    n_features=10,
    centers=4,
    cluster_std=2.0,
    random_state=42
)

feature_names = [f"Feature_{i+1}" for i in range(10)]

df = pd.DataFrame(X, columns=feature_names)
df["True_Demo_Cluster"] = true_labels

df.to_csv(
    "data/raw/synthetic_clustering_dataset.csv",
    index=False
)

print("Synthetic dataset created:", df.shape)

# ============================================================
# NOTEBOOK
# ============================================================

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True)
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True)
    })

md("""# Dimensionality Reduction & Unsupervised Clustering

## Objective

Apply Principal Component Analysis (PCA) and unsupervised clustering algorithms to discover patterns in high-dimensional numerical data.

### Methods
- Standardization
- PCA
- Explained Variance Ratio
- Scree Plot
- K-Means
- Elbow Method
- Silhouette Score
- DBSCAN
- Hierarchical Clustering
- 2D PCA Cluster Visualizations

**Data note:** This implementation uses a clearly labeled synthetic dataset because no real dataset was provided. The results demonstrate the methodology and reproducibility of the workflow rather than real-world findings.
""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

np.random.seed(42)

X, true_labels = make_blobs(
    n_samples=600,
    n_features=10,
    centers=4,
    cluster_std=2.0,
    random_state=42
)

feature_names = [f"Feature_{i+1}" for i in range(10)]

df = pd.DataFrame(X, columns=feature_names)

print("Dataset shape:", df.shape)
display(df.head())
""")

md("""## 1. Standardize High-Dimensional Numerical Features""")

code("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names
)

display(scaled_df.describe().round(3))
""")

md("""## 2. Principal Component Analysis (PCA)""")

code("""pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

variance_table = pd.DataFrame({
    "Principal Component": np.arange(1, 11),
    "Explained Variance Ratio": explained_variance,
    "Cumulative Variance": cumulative_variance
})

display(variance_table.round(4))

n_components_90 = np.argmax(
    cumulative_variance >= 0.90
) + 1

print(
    "Components required for at least 90% variance:",
    n_components_90
)
""")

md("""## 3. PCA Scree Plot""")

code("""plt.figure(figsize=(9,5))

plt.plot(
    range(1, 11),
    explained_variance,
    marker="o"
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
""")

md("""## 4. Cumulative Explained Variance""")

code("""plt.figure(figsize=(9,5))

plt.plot(
    range(1, 11),
    cumulative_variance,
    marker="o"
)

plt.axhline(
    0.90,
    linestyle="--",
    label="90% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.show()
""")

md("""## 5. Two-Dimensional PCA Projection""")

code("""pca_2 = PCA(n_components=2)

X_pca = pca_2.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

print(
    "PC1 variance:",
    round(pca_2.explained_variance_ratio_[0], 4)
)

print(
    "PC2 variance:",
    round(pca_2.explained_variance_ratio_[1], 4)
)

print(
    "Total 2D variance:",
    round(pca_2.explained_variance_ratio_.sum(), 4)
)
""")

md("""## 6. K-Means — Elbow Method""")

code("""inertias = []

k_values = range(2, 11)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

plt.figure(figsize=(9,5))

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")
plt.grid(True)
plt.show()
""")

md("""## 7. K-Means — Silhouette Score Analysis""")

code("""silhouette_results = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append(score)

silhouette_table = pd.DataFrame({
    "k": range(2, 11),
    "Silhouette Score": silhouette_results
})

display(
    silhouette_table.round(4)
)

best_k = int(
    silhouette_table.loc[
        silhouette_table["Silhouette Score"].idxmax(),
        "k"
    ]
)

print(
    "Best k according to silhouette score:",
    best_k
)
""")

code("""kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print(
    "K-Means silhouette score:",
    round(kmeans_score, 4)
)
""")

md("""## 8. K-Means Cluster Visualization""")

code("""plot_kmeans = pca_df.copy()

plot_kmeans["Cluster"] = (
    kmeans_labels.astype(str)
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=plot_kmeans,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=60
)

plt.title(
    "K-Means Clusters — PCA 2D Projection"
)

plt.grid(True)
plt.show()
""")

md("""## 9. DBSCAN Clustering""")

code("""dbscan = DBSCAN(
    eps=1.2,
    min_samples=8
)

dbscan_labels = dbscan.fit_predict(X_scaled)

n_dbscan_clusters = (
    len(set(dbscan_labels))
    - (1 if -1 in dbscan_labels else 0)
)

n_noise = int(
    np.sum(dbscan_labels == -1)
)

print(
    "DBSCAN clusters:",
    n_dbscan_clusters
)

print(
    "DBSCAN noise points:",
    n_noise
)

valid_dbscan = dbscan_labels != -1

if (
    valid_dbscan.sum() > 1
    and len(set(dbscan_labels[valid_dbscan])) >= 2
):

    dbscan_silhouette = silhouette_score(
        X_scaled[valid_dbscan],
        dbscan_labels[valid_dbscan]
    )

    print(
        "DBSCAN silhouette score:",
        round(dbscan_silhouette, 4)
    )

else:

    dbscan_silhouette = None

    print(
        "DBSCAN silhouette score: Not available"
    )
""")

md("""## 10. DBSCAN Cluster Visualization""")

code("""plot_dbscan = pca_df.copy()

plot_dbscan["Cluster"] = (
    dbscan_labels.astype(str)
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=plot_dbscan,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=60
)

plt.title(
    "DBSCAN Clusters — PCA 2D Projection"
)

plt.grid(True)
plt.show()
""")

md("""## 11. Hierarchical Clustering""")

code("""hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = (
    hierarchical.fit_predict(X_scaled)
)

hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print(
    "Hierarchical clustering silhouette score:",
    round(hierarchical_silhouette, 4)
)
""")

md("""## 12. Hierarchical Cluster Visualization""")

code("""plot_hierarchical = pca_df.copy()

plot_hierarchical["Cluster"] = (
    hierarchical_labels.astype(str)
)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=plot_hierarchical,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=60
)

plt.title(
    "Hierarchical Clusters — PCA 2D Projection"
)

plt.grid(True)
plt.show()
""")

md("""## 13. Clustering Comparison""")

code("""comparison = pd.DataFrame({
    "Method": [
        "K-Means",
        "Hierarchical"
    ],
    "Clusters": [
        len(np.unique(kmeans_labels)),
        len(np.unique(hierarchical_labels))
    ],
    "Silhouette Score": [
        kmeans_score,
        hierarchical_silhouette
    ]
})

display(
    comparison.round(4)
)

if dbscan_silhouette is not None:

    print(
        "DBSCAN silhouette score:",
        round(dbscan_silhouette, 4)
    )
""")

md("""# Findings

1. The numerical features were standardized before dimensionality reduction and clustering.
2. PCA reduced the high-dimensional feature space while providing explained-variance information.
3. The scree plot and cumulative explained variance plot were generated to support PCA component selection.
4. K-Means was evaluated using both the Elbow Method and Silhouette Score.
5. DBSCAN was applied as a density-based clustering method and potential noise observations were identified.
6. Hierarchical clustering was applied as an additional unsupervised approach.
7. PCA 2D projections were used to visually inspect the discovered cluster structures.

**Important:** The dataset is synthetic and the findings demonstrate the analytical workflow rather than describing a real-world population.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": sys.version.split()[0]
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

notebook_path = (
    "notebooks/dimensionality_reduction_clustering.ipynb"
)

with open(
    notebook_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        notebook,
        f,
        indent=2
    )

# ============================================================
# EXECUTE NOTEBOOK
# ============================================================

import nbformat
from nbclient import NotebookClient

with open(
    notebook_path,
    "r",
    encoding="utf-8"
) as f:

    nb = nbformat.read(
        f,
        as_version=4
    )

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3"
)

client.execute()

with open(
    notebook_path,
    "w",
    encoding="utf-8"
) as f:

    nbformat.write(
        nb,
        f
    )

# ============================================================
# REPORT
# ============================================================

report = """# Dimensionality Reduction & Unsupervised Clustering

## Objective

Apply PCA and unsupervised clustering algorithms to discover patterns in high-dimensional numerical data.

## Dataset

A reproducible synthetic dataset containing 600 observations and 10 numerical features was generated using a fixed random seed.

## Techniques Applied

- StandardScaler
- Principal Component Analysis (PCA)
- Explained Variance Ratio
- Scree Plot
- Cumulative Explained Variance
- K-Means
- Elbow Method
- Silhouette Score
- DBSCAN
- Hierarchical/Agglomerative Clustering
- 2D PCA Cluster Visualizations

## Reproducibility

Random seed: `42`

The complete executable workflow is available in:

`notebooks/dimensionality_reduction_clustering.ipynb`

## Data Limitation

The dataset is synthetic because no real dataset was provided. Therefore, the clustering results demonstrate the implementation methodology and should not be interpreted as real-world evidence.
"""

with open(
    "reports/dimensionality_reduction_clustering_report.md",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ============================================================
# GIT PUSH
# ============================================================

subprocess.run(
    ["git", "add", "."],
    check=True
)

check = subprocess.run(
    ["git", "diff", "--cached", "--quiet"]
)

if check.returncode != 0:

    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Add PCA and unsupervised clustering analysis"
        ],
        check=True
    )

    subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        check=True
    )

else:

    print("No new Git changes to commit.")

print("\n" + "=" * 60)
print("TASK COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nNotebook:")
print(
    "notebooks\\dimensionality_reduction_clustering.ipynb"
)

print("\nDataset:")
print(
    "data\\raw\\synthetic_clustering_dataset.csv"
)

print("\nReport:")
print(
    "reports\\dimensionality_reduction_clustering_report.md"
)

print("\nGitHub:")
print(
    "https://github.com/Maddy67bh/reproducible-research"
)

print("\nGit Status:")
subprocess.run(
    ["git", "status", "--short"]
)
