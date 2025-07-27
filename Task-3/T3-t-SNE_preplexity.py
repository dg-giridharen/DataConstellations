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

perplexity_values = [5, 15, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('t-SNE with Different Perplexity Values', fontsize=16)
plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

sns.set_theme(style="whitegrid", palette="viridis")

for perplexity, pos in zip(perplexity_values, plot_positions):
    tsne_transformer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        init='pca',
        random_state=42
    )
    tsne_result = tsne_transformer.fit_transform(scaled_features)

    ax = axes[pos]
    sns.scatterplot(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        hue=target_names[labels],
        ax=ax
    )
    ax.set_title(f'Perplexity = {perplexity}')
    ax.legend([], [], frameon=False)

handles, legend_labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()