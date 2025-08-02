# 🧠 Big Five Personality Clustering & Classification Project

## 📌 Project Overview
This project focuses on analyzing and classifying psychological profiles using responses to the **Big Five Personality Test**.  
The methodology integrates **unsupervised learning** (for cluster discovery) and **supervised learning** (for classification), followed by an **ensemble approach** to maximize predictive performance.

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
Clusters were characterized by percentile analysis of OCEAN traits:

| Cluster | Name        | Description |
|---------|-------------|-------------|
| 0       | **Rebels**   | Medium Openness, Low Conscientiousness, Medium Extraversion, Low Agreeableness, Medium Neuroticism |
| 1       | **Leaders**  | High Openness, High Conscientiousness, High Extraversion, High Agreeableness, Medium Neuroticism |
| 2       | **Anxious**  | Low Openness, Medium Conscientiousness, Low Extraversion, Medium Agreeableness, High Neuroticism |
| 3       | **Balanced** | Medium Openness, Medium Conscientiousness, Medium Extraversion, Medium Agreeableness, Low Neuroticism |

---

## 🔥 Supervised Learning

Once clusters were identified, the problem was reframed as a **4-class classification** task.

### ✅ **Models Tested**
1. **XGBoost**
2. **Multi-Layer Perceptron (MLP)** (TensorFlow/Keras)

Both models were trained first with default parameters, then optimized with **Optuna** for hyperparameter tuning.

---

## 🎯 Results Summary

| Model                     | Tuning        | Accuracy | F1-Score |
|---------------------------|---------------|----------|----------|
| XGBoost                   | No tuning     | 0.889    | 0.890    |
| XGBoost (Optuna)          | ✅ Optuna     | 0.914    | 0.914    |
| MLP                       | No tuning     | 0.922    | 0.921    |
| MLP (Optuna)              | ✅ Optuna     | 0.937    | 0.937    |
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
1. **Loads the optimized models** (`xgb_optuna.pkl` and `mlp_tuning.h5`)
2. **Loads the dataset** (`big5_supervised_dataset.csv`) and splits it into train/test
3. **Generates predictions** using both XGBoost and MLP
4. **Applies soft voting** with the optimal weights (0.3 for XGB, 0.7 for MLP)
5. **Evaluates** performance (accuracy, F1-score, classification report)
6. **Plots** the confusion matrix
7. **Saves predictions** and class probabilities for further analysis

> ✅ This makes it easy to **reproduce the final results** without re-running the entire training and tuning process.

---

## ✅ **Conclusions**
- **GMM clustering** successfully discovered 4 meaningful personality profiles.  
- **MLP Optuna** outperformed XGBoost as a standalone classifier.  
- **Soft Voting Ensemble** achieved the **best performance**, confirming that combining models increases robustness and accuracy.  
- Final model achieves **94% accuracy**, making it a strong predictive solution for personality classification.

---

## 🚀 Future Work
- Apply **SHAP** analysis to interpret which features most influence predictions.  
- Explore other ensemble techniques (stacking, blending).  
- Test alternative clustering algorithms (e.g., HDBSCAN) to validate cluster structure.  
- Deploy the final ensemble model as an **interactive web app**.

---

## 🛠️ Technologies Used
- **Python**, **scikit-learn**, **Optuna**, **TensorFlow/Keras**, **XGBoost**
- **Matplotlib**, **Pandas**, **NumPy**

---

## 👤 Author
Developed by **Samuele** as part of an advanced machine learning exploration on psychological profiling.
