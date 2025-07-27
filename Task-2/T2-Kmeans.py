import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = 'Banana.jpeg'

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image_rgb.reshape(-1, 3).astype(np.float32)

k = 4
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
kmeans.fit(pixels)

new_colors = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = new_colors.reshape(image_rgb.shape).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(segmented_image)
axes[1].set_title(f'Segmented Image (k={k})')
axes[1].axis('off')

plt.tight_layout()
plt.show()