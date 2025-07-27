import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

image_path = 'banana.jpeg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixels = image_rgb.reshape(-1, 3).astype(np.float32)

k = 3
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
kmeans.fit(pixels)

dominant_color_cluster_index = np.argmax(np.bincount(kmeans.labels_))
dominant_color = kmeans.cluster_centers_[dominant_color_cluster_index]

mask = np.zeros_like(kmeans.labels_, dtype=np.uint8)
mask[kmeans.labels_ == dominant_color_cluster_index] = 255
mask = mask.reshape(image_rgb.shape[:2])

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output_image = image_rgb.copy()
if contours:
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Object Mask')
axes[1].axis('off')

axes[2].imshow(output_image)
axes[2].set_title('Image with Bounding Box')
axes[2].axis('off')

plt.tight_layout()
plt.show()