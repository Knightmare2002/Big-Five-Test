# 🧠 Big Five Personality Clustering & Classification Project

## 📌 Project Overview
This project focuses on analyzing and classifying psychological profiles using responses to the **Big Five Personality Test**.  
The methodology integrates **unsupervised learning** (for cluster discovery) and **supervised learning** (for classification), followed by an **ensemble approach** to maximize predictive performance.  
Additionally, an **interactive Streamlit web application** was developed to allow users to take the Big Five test and receive an ML-powered personality profile.

---

## 📂 Dataset
- **Source**: OpenPsychometrics Big Five open dataset  
- **Features**: 50 item-level responses (Likert scale)  
- **Target**: Cluster labels derived from Gaussian Mixture Model (GMM) clustering  

---

## 🧩 Unsupervised Learning (GMM)

### ✅ **Clustering Process**
- Applied **PCA** for dimensionality reduction (90% explained variance).  
- Used **Gaussian Mixture Models (GMM)** with full covariance.  
- Model selection based on **AIC**, **BIC**, and **Silhouette Score**.  
- Optimal number of clusters: **4**.

### ✅ **Cluster Profiles**
The final clusters were manually labeled based on their OCEAN centroids:

| Cluster | Name             | Description |
|---------|------------------|-------------|
| 0       | **Reserved**     | Moderate Openness, Low Agreeableness and Conscientiousness, slightly elevated Neuroticism |
| 1       | **Striver**      | Moderate-to-high O, C, E, A, with elevated Neuroticism (driven achievers but emotionally tense) |
| 2       | **Internalizer** | Low O and E, moderate A, very high Neuroticism (introverted and emotionally vulnerable) |
| 3       | **Balanced**     | Average O, C, E, A with very low Neuroticism (emotionally stable and well-adjusted) |

---

## 🔥 Supervised Learning

Once clusters were identified, the problem was reframed as a **4-class classification** task.

### ✅ **Models Tested**
1. **XGBoost**
2. **Multi-Layer Perceptron (MLP)** (TensorFlow/Keras)

Both models were trained first with default parameters, then optimized with **Optuna** for hyperparameter tuning.

---

## 🎯 Results Summary

| Model                       | Tuning        | Accuracy | F1-Score |
|-----------------------------|---------------|----------|----------|
| XGBoost                     | No tuning     | 0.889    | 0.890    |
| XGBoost (Optuna)            | ✅ Optuna     | 0.914    | 0.914    |
| MLP                         | No tuning     | 0.922    | 0.921    |
| MLP (Optuna)                | ✅ Optuna     | 0.937    | 0.937    |
| **Soft Voting (XGB + MLP)** | ✅ Ensemble   | **0.940**| **0.940**|

---

## 🤝 Ensemble Learning (Soft Voting)

### ✅ **Approach**
- Combined the **probabilities** predicted by XGBoost and MLP using a **weighted average**.
- Grid search was used to find optimal weights maximizing the F1-score.

### ✅ **Optimal Weights**
- **XGB weight**: 0.3  
- **MLP weight**: 0.7  

### ✅ **Performance**
- **Accuracy: 0.9401**  
- **F1-score: 0.9401**  

---

## 🖥️ `final_pipeline.py`

The script **`final_pipeline.py`** is a **ready-to-run inference pipeline** that:
1. **Loads** the optimized models (`xgb_optuna.pkl` and `mlp_tuning.h5`)
2. **Loads** the dataset (`big5_supervised_dataset.csv`) and splits it into train/test
3. **Generates predictions** using both XGBoost and MLP
4. **Applies soft voting** with optimal weights
5. **Evaluates** model performance and plots the confusion matrix
6. **Saves** predictions and class probabilities for further analysis

---

## 🌐 Streamlit Web App (`big5_app/`)

An **interactive Streamlit web app** was implemented, allowing users to:
- **Take the Big Five Personality Test** directly in the browser (50 questions)  
- **Get predictions** from the trained ensemble model (XGB + MLP)  
- **Visualize results** via:
  - Class probabilities bar chart
  - Radar chart of OCEAN traits  
  - Display of predicted psychological profile

The app was structured to run both **locally** and can be deployed online (e.g., via **Streamlit Cloud** or **Heroku**) without requiring Python installation by the user.

---

## ✅ **Conclusions**
- **GMM clustering** successfully identified 4 distinct psychological profiles.  
- **MLP Optuna** achieved the best individual performance.  
- **Soft Voting Ensemble** further improved accuracy and robustness.  
- The addition of the **Streamlit app** enables interactive personality testing using the trained models.

---

## 🚀 Future Work
- Extend dataset with cross-cultural samples.  
- Explore **SHAP** for explainability.  
- Test additional ensemble methods (stacking, blending).  
- Deploy the app to a cloud service for public access.  

---

## 🛠️ Technologies Used
- **Python**, **scikit-learn**, **Optuna**, **TensorFlow/Keras**, **XGBoost**, **Streamlit**
- **Matplotlib**, **Pandas**, **NumPy**

---

## 👤 Author
Developed by **Samuele** as part of an advanced machine learning project on psychological profiling.
