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

X_accel = X_train.iloc[:, 0:3]
X_gyro = X_train.iloc[:, 3:6]
X_combined = X_train.iloc[:, 0:6]

feature_sets = {
    'Acceleration Only': X_accel,
    'Gyroscope Only': X_gyro,
    'Combined Accel & Gyro': X_combined
}

fig, axes = plt.subplots(len(feature_sets), 2, figsize=(12, 15))
fig.suptitle('Clustering Quality with Different Feature Sets', fontsize=16)

for i, (title, X_data) in enumerate(feature_sets.items()):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    predicted_labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    axes[i, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=predicted_labels, cmap='viridis', s=10)
    axes[i, 0].set_title(f'K-Means on {title}')
    axes[i, 0].set_ylabel('PC 2')
    if i == len(feature_sets) - 1:
      axes[i, 0].set_xlabel('PC 1')


    axes[i, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_train['activity'], cmap='viridis', s=10)
    axes[i, 1].set_title(f'True Labels on {title}')
    if i == len(feature_sets) - 1:
      axes[i, 1].set_xlabel('PC 1')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()