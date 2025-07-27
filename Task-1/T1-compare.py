import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

def create_spiral_dataset(points_per_arm=100, num_arms=3, noise_factor=0.5):
    all_features = []
    for arm_index in range(num_arms):
        radius = np.linspace(0.0, 1.0, points_per_arm)
        angle = np.linspace(arm_index * 4, (arm_index + 1) * 4, points_per_arm) \
                + np.random.randn(points_per_arm) * noise_factor
        feature_x = radius * np.sin(angle)
        feature_y = radius * np.cos(angle)
        all_features.extend(zip(feature_x, feature_y))
    return np.array(all_features)

features = create_spiral_dataset(points_per_arm=100, num_arms=3)

kmeans_labels = KMeans(n_clusters=3, n_init='auto', random_state=42).fit_predict(features)
dbscan_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(features)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Algorithm Comparison on Spiral Dataset', fontsize=16)

axes[0].scatter(features[:, 0], features[:, 1], c=kmeans_labels, cmap='viridis', s=40)
axes[0].set_title('K-Means Result')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[1].scatter(features[:, 0], features[:, 1], c=dbscan_labels, cmap='viridis', s=40)
axes[1].set_title('DBSCAN Result')
axes[1].set_xlabel("Feature 1")
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()