import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def create_spiral_dataset(points_per_arm=100, num_arms=3, noise_factor=0.5):
    all_features = []
    all_labels = []

    for arm_index in range(num_arms):
        radius = np.linspace(0.0, 1.0, points_per_arm)
        angle = np.linspace(arm_index * 4, (arm_index + 1) * 4, points_per_arm) \
                + np.random.randn(points_per_arm) * noise_factor

        feature_x = radius * np.sin(angle)
        feature_y = radius * np.cos(angle)

        all_features.extend(zip(feature_x, feature_y))
        all_labels.extend([arm_index] * points_per_arm)

    return np.array(all_features), np.array(all_labels)

if __name__ == '__main__':
    spiral_features, spiral_labels = create_spiral_dataset(points_per_arm=100, num_arms=3)
    data_to_save = np.hstack((spiral_features, spiral_labels.reshape(-1, 1)))
    
    df = pd.DataFrame(data_to_save, columns=['feature_1', 'feature_2', 'cluster_label'])

    file_path = 'spiral_dataset.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Dataset successfully saved to: {file_path}")

    plt.figure(figsize=(8, 6))
    plt.title("Colorful Spiral Dataset")
    plt.scatter(spiral_features[:, 0], spiral_features[:, 1], c=spiral_labels, cmap='viridis', edgecolor='k', s=40)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()