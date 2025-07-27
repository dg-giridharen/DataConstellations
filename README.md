# 🧠 Unsupervised Machine Learning Case Study

This repository contains a collection of completed tasks for a machine learning skills assessment. The project focuses on applying various **unsupervised learning techniques** to analyze, cluster, and visualize different types of data.

---

## 📂 Project Tasks

This project is divided into six distinct tasks, each exploring a different aspect of unsupervised learning.

---

### ✅ Task 1: Clustering Fundamentals

- Compared **K-Means** and **DBSCAN** clustering algorithms.
- Applied them on both **globular** and **arbitrarily shaped** synthetic datasets (e.g., spiral).
- Analyzed how K-Means struggles with non-convex clusters while DBSCAN performs well.

---

### 🖼️ Task 2: Image Segmentation with K-Means

- Performed **color quantization** using K-Means on an image.
- Reduced the number of pixel colors to create a **segmented/posterized** effect.
- Conducted experiments in both **RGB** and **LAB** color spaces.

---

### 📉 Task 3: Dimensionality Reduction

- Applied **PCA** and **t-SNE** to high-dimensional datasets (e.g., Iris, Digits).
- Reduced to 2D for visualization.
- Compared PCA’s linear transformation with t-SNE’s non-linear mapping.
- Highlighted t-SNE's superior ability to form **distinct clusters** visually.

---

### 📡 Task 4: Real-World Sensor Data Clustering

- Used the **UCI Human Activity Recognition (HAR)** dataset.
- Applied **K-Means** clustering to sensor data (accelerometer and gyroscope).
- Analyzed how unsupervised methods can group real-world human activities like **walking**, **sitting**, and **standing**.

---

### 🔍 Task 5: Object Outlining from Segmentation

- Built on Task 2’s segmented image.
- Identified the main object cluster using K-Means.
- Used **OpenCV** to find contours and draw bounding boxes.
- Simulated a basic **computer vision/AR application**.

---

### 🧩 Task 6: Implementing an Existing GitHub Project

- Cloned and ran an open-source **image segmentation** project.
- Explored existing codebases to understand and analyze segmentation methods.
- Demonstrated ability to work with real-world open-source projects.

---

## 📊 Visual Results

### 🔄 K-Means vs. DBSCAN on Spiral Dataset

- ✅ DBSCAN successfully clustered the spiral shape.
- ❌ K-Means failed due to its assumption of convex clusters.

### 🟡 K-Means Image Segmentation

- Segmenting a **banana image** using `k=4` resulted in a posterized effect.
- Shows the power of unsupervised methods in computer vision tasks.

---

## 🧰 Technologies Used

- **Python**
- `scikit-learn` – Clustering, PCA, t-SNE
- `pandas` – Data manipulation
- `NumPy` – Numerical operations
- `Matplotlib`, `Seaborn` – Data visualization
- `OpenCV` – Image processing
- `Jupyter Notebook` – Interactive development

---


