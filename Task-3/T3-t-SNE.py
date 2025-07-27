import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

iris = load_iris()
features = iris.data
labels = iris.target
target_names = iris.target_names

scaled_features = StandardScaler().fit_transform(features)

tsne_transformer = TSNE(n_components=2, perplexity=15, learning_rate='auto', init='pca', random_state=42)
tsne_result = tsne_transformer.fit_transform(scaled_features)

sns.set_theme(style="whitegrid", palette="viridis")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle('t-SNE of Iris Dataset', fontsize=16)

sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    hue=target_names[labels],
    ax=ax
)
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()