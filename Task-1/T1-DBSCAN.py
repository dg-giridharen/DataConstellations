import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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

spiral_features = create_spiral_dataset(points_per_arm=100, num_arms=3)

dbscan_model = DBSCAN(eps=0.2, min_samples=5)
predicted_labels = dbscan_model.fit_predict(spiral_features)

plt.figure(figsize=(8, 6))
plt.scatter(
    spiral_features[:, 0],
    spiral_features[:, 1],
    c=predicted_labels,
    cmap='viridis',
    s=40
)

plt.title('DBSCAN Clustering on Spiral Dataset')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()