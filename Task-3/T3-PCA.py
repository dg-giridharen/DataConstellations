import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
features = iris.data
labels = iris.target
target_names = iris.target_names

scaled_features = StandardScaler().fit_transform(features)

pca_transformer = PCA(n_components=2, random_state=42)
pca_result = pca_transformer.fit_transform(scaled_features)

sns.set_theme(style="whitegrid", palette="viridis")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle('PCA of Iris Dataset', fontsize=16)

sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=target_names[labels],
    ax=ax
)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()