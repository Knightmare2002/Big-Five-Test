# 🧠 Big Five Personality Clustering Project

## 📌 Project Overview
This project aims to analyze and cluster individuals based on their responses to the **Big Five Personality Test** (50 items, measuring Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism – OCEAN).  
The final goal is to create:
1. **Meaningful psychological profiles** using unsupervised learning (Gaussian Mixture Models – GMM).
2. A **labeled dataset** suitable for supervised learning and deep learning applications.

---

## 🗂 Dataset
- Source: **OpenPsychometrics – Big Five dataset**
- Contains **50 item-level features** (`E1..E10`, `O1..O10`, `C1..C10`, `A1..A10`, `N1..N10`), each representing a response to a personality test question.
- Preprocessing:
  - Items were used directly (standardized when needed).
  - Additional aggregated scores for O, C, E, A, N were computed for analysis.

---

## 🔍 Exploratory Data Analysis (EDA)
1. **PCA (Principal Component Analysis)** was applied to:
   - Visualize the shape of the data.
   - Determine the optimal number of components for dimensionality reduction.
2. **Findings:**
   - For item-level data, **37 components** were required to explain 90% of the variance.
   - A **2D PCA projection** showed a **dense, elliptical, and continuous** distribution — consistent with the theory that personality traits are continuous rather than discrete.

---

## 🤖 Clustering Approach

### 1. **K-Means**
- Tried initially, but silhouette scores were **very low (~0.1-0.2)**.
- Confirmed that the dataset lacks clear, spherical clusters.
- Conclusion: **Not suitable**.

### 2. **Gaussian Mixture Models (GMM)**
- GMM was chosen because it:
  - Models overlapping elliptical clusters.
  - Provides probabilistic cluster assignments.
  - Supports model selection using **AIC** and **BIC**.

---

## 📈 Model Evaluation
- The number of Gaussian components was varied (`n_components` ∈ [1, 6]).
- **Evaluation metrics:**
  - **AIC/BIC**: To select the model with the best trade-off between fit and complexity.
  - **Silhouette Score**: To measure separation between clusters (values remained low, confirming overlap).

### **Findings:**
- **Using PCA with 6 components**:
  - BIC suggested **3–4 clusters** as optimal.
  - AIC favored **6 clusters**, but with high overlap.
- **Using PCA with 37 components**:
  - BIC plateaued after **3 components**.
  - Silhouette scores became **negative** for high cluster counts → indicating strong overlap.

---

## 🎯 Selected Model
- Final chosen configuration:  
  - **GMM with 4 Gaussian components** (good compromise between interpretability and fit).
  - Visualization with covariance ellipses confirmed significant **overlap**, but identifiable density regions.

---

## 🧩 Cluster Analysis

Cluster means were calculated using the aggregated OCEAN scores. Percentile thresholds were used to label traits as **Low**, **Medium**, or **High**.  

### 📌 **Cluster Profiles**

| Cluster | O | C | E | A | N | Description |
|---------|---|---|---|---|---|-------------|
| **0** | Medium | Low | Medium | Low | Medium | **Rebellious Balanced** – moderately sociable, emotionally average, less disciplined and cooperative. |
| **1** | High | High | High | High | Medium | **Well-Adjusted Achievers** – open, conscientious, sociable, cooperative, emotionally balanced. |
| **2** | Low | Medium | Low | Medium | High | **Anxious Introverts** – closed to experiences, introverted, anxious, moderately agreeable. |
| **3** | Medium | Medium | Medium | Medium | Low | **Calm Balanced** – emotionally stable, with moderate traits across all dimensions. |

---

## 🗡️ Compact Labels for ML/DL
For supervised/deep learning applications, compact class names were also defined:
- `0 → Rebels`
- `1 → Leaders`
- `2 → Anxious`
- `3 → Balanced`

---

## 📊 Visualization
- **PCA 2D Scatter Plots** with GMM ellipses clearly showed overlapping but distinguishable density regions.
- **Radar Charts** were used to compare average OCEAN profiles across clusters.

---

## 📦 Dataset for Supervised Learning
A final dataset was created:
- Contains all **50 item-level features**.
- Includes a **Cluster** column (GMM labels).
- Exported as `big5_supervised_dataset.csv` for downstream ML/DL tasks.

---

## ✅ Key Insights
- The dataset exhibits **continuous personality distributions**, not clear-cut clusters.
- GMM successfully identified **4 meaningful psychological profiles**, aligning with Big Five theory.
- This labeled dataset can now serve as the basis for **classification models** (e.g., neural networks, XGBoost).

---

## 🚀 Next Steps
1. Train **supervised models** to predict personality cluster from raw item responses.
2. Experiment with **deep learning architectures** (MLP, autoencoders).
3. Compare with other unsupervised approaches (Hierarchical Clustering, DBSCAN, Variational Bayesian GMM).

---

## 🏁 Conclusion
This project demonstrates how **unsupervised learning (GMM)** can uncover meaningful patterns in personality data, even when cluster separation is low. The resulting **cluster labels** offer a psychologically interpretable structure and a foundation for supervised learning tasks.

---
