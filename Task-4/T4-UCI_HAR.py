import os
import urllib.request
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
zip_filename = 'UCI_HAR_Dataset.zip'
dataset_folder = 'UCI HAR Dataset'

if not os.path.exists(dataset_folder):
    print('Dataset not found. Downloading...')
    urllib.request.urlretrieve(zip_url, zip_filename)
    print('Download complete. Extracting files...')
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall()
    print('Extraction complete.')
    os.remove(zip_filename)
else:
    print('Dataset already exists.')

features_path = os.path.join(dataset_folder, 'train', 'X_train.txt')
labels_path = os.path.join(dataset_folder, 'train', 'y_train.txt')

X_train = pd.read_csv(features_path, sep='\s+', header=None)
y_train = pd.read_csv(labels_path, sep='\s+', header=None, names=['activity'])

X_subset = X_train.head(2000)
y_subset = y_train.head(2000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
predicted_labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Clustering Sensor Data vs. True Labels (via PCA)', fontsize=16)

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_labels, cmap='viridis', s=15, alpha=0.7)
axes[0].set_title('Clusters Found by K-Means')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_subset['activity'], cmap='viridis', s=15, alpha=0.7)
axes[1].set_title('Actual Activity Labels')
axes[1].set_xlabel('Principal Component 1')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()