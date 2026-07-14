from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# K-MEANS CLUSTERING
# ===============================

# Use only features

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

df["Cluster"] = clusters

print(df.head())

# Visualize Clusters

if X.shape[1] >= 2:
    plt.figure(figsize=(8,6))

    plt.scatter(
        X.iloc[:,0],
        X.iloc[:,1],
        c=clusters,
        cmap="viridis"
    )

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("K-Means Clustering")
    plt.show()
