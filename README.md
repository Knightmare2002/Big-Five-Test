# 🧠 Big Five Personality Clustering & Classification Project

## 📌 Project Overview
This project aims to **analyze and classify psychological profiles** based on responses to the Big Five Personality Test (OCEAN model: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

The pipeline consists of:
1. **Data preprocessing** of Big Five questionnaire data
2. **Dimensionality reduction** with PCA
3. **Unsupervised clustering** using Gaussian Mixture Models (GMM)
4. **Cluster analysis and labeling** based on OCEAN trait distributions
5. **Supervised learning models** (XGBoost & MLP) trained to predict cluster membership
6. **Hyperparameter tuning** using Optuna for optimal performance

---

## 📂 Dataset
- Source: **OpenPsychometrics** Big Five open dataset
- Features: **50 items** (Likert scale responses)
- Target: **Cluster labels** obtained from GMM

---

## 🧩 Unsupervised Learning (GMM)

### ✅ **Clustering Approach**
- Applied PCA to reduce dimensionality while retaining 90% of variance.
- Tested multiple GMM configurations using:
  - **AIC** and **BIC** for model selection
  - **Silhouette Score** for cluster quality evaluation
- Optimal number of clusters: **4**

### ✅ **Cluster Profiles**
Based on percentile analysis of OCEAN scores:

| Cluster | Name        | Description |
|---------|-------------|-------------|
| 0       | **Rebels**   | Medium Openness, Low Conscientiousness, Medium Extraversion, Low Agreeableness, Medium Neuroticism |
| 1       | **Leaders**  | High Openness, High Conscientiousness, High Extraversion, High Agreeableness, Medium Neuroticism |
| 2       | **Anxious**  | Low Openness, Medium Conscientiousness, Low Extraversion, Medium Agreeableness, High Neuroticism |
| 3       | **Balanced** | Medium Openness, Medium Conscientiousness, Medium Extraversion, Medium Agreeableness, Low Neuroticism |

---

## 🔥 Supervised Learning

After defining the clusters, the problem was reframed as a **multiclass classification** task.

### ✅ **Models Tested**
1. **XGBoost**
2. **Multi-Layer Perceptron (MLP)**

---

## 🎯 Results Summary

| Model                | Tuning        | Accuracy | F1-Score |
|----------------------|--------------|----------|----------|
| XGBoost              | No tuning    | 0.889    | 0.890    |
| XGBoost (Optuna)     | ✅ Optuna    | 0.914    | 0.914    |
| MLP                  | No tuning    | 0.922    | 0.921    |
| MLP (Optuna)         | ✅ Optuna    | **0.937**| **0.937**|

---

## ✅ **Conclusions**
- **GMM clustering** revealed 4 meaningful personality clusters.
- **XGBoost** performed well, achieving **91% accuracy** after tuning.
- **MLP** outperformed XGBoost, with **94% accuracy** after Optuna tuning.
- **Optuna** was crucial in significantly improving model performance, especially for the MLP.

---

## 🚀 Future Work
- Apply **SHAP** analysis for feature interpretability (which items drive predictions?).
- Explore **ensemble models** combining XGBoost and MLP.
- Try alternative clustering methods (HDBSCAN, Spectral Clustering) to validate cluster consistency.
- Deploy the model as an **API** or interactive **web app**.

---

## 🛠️ Technologies Used
- **Python**, **scikit-learn**, **Optuna**, **TensorFlow/Keras**, **XGBoost**
- **Matplotlib**, **Pandas**, **NumPy**

---

## 👤 Author
Project developed by **Samuele** as part of an advanced ML experimentation on psychological data.

