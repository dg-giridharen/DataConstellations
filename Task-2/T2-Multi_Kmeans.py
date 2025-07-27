import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = 'Banana.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image_rgb.reshape(-1, 3).astype(np.float32)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Image Segmentation with Different k Values', fontsize=16)

k_values_to_test = [2, 4, 8, 16]
plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for k, pos in zip(k_values_to_test, plot_positions):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(pixels)
    
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = new_colors.reshape(image_rgb.shape).astype(np.uint8)
    
    ax = axes[pos]
    ax.imshow(segmented_image)
    ax.set_title(f'k = {k}')
    ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()