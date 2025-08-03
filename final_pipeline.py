import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ================================
# 🔹 1. Load Models
# ================================
print("🔹 Loading optimized models...")
xgb_model = joblib.load("xgb_optuna.pkl")
mlp_model = load_model("mlp_tuning.h5")

# ================================
# 🔹 2. Load Data 
# ================================
df_supervised = pd.read_csv("big5_supervised_dataset.csv")

X = df_supervised.loc[:, "E1":"O10"].copy()
y = df_supervised["Cluster"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================================
# 🔹 3. Predictions (Probabilities)
# ================================
print("🔹 Generating predictions...")
proba_xgb = xgb_model.predict_proba(X_test)
proba_mlp = mlp_model.predict(X_test)

# ================================
# 🔹 4. Ensemble (Soft Voting)
# ================================
w_xgb = 0.3
w_mlp = 0.7
final_proba = w_xgb * proba_xgb + w_mlp * proba_mlp
y_pred_ensemble = np.argmax(final_proba, axis=1)

# ================================
# 🔹 5. Evaluation
# ================================
print("\n=== Ensemble Model Performance ===")
acc = accuracy_score(y_test, y_pred_ensemble)
f1 = f1_score(y_test, y_pred_ensemble, average="weighted")
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(classification_report(y_test, y_pred_ensemble))

# ================================
# 🔹 6. Confusion Matrix
# ================================
print("🔹 Plotting confusion matrix...")
cluster_names = {
    0: "Reserved",
    1: "Striver",
    2: "Internalizer",
    3: "Balanced"
}

cm = confusion_matrix(y_test, y_pred_ensemble)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[cluster_names[i] for i in sorted(cluster_names.keys())])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Soft Voting Ensemble")
plt.show()

# ================================
# 🔹 7. Save Predictions
# ================================
np.save("ensemble_predictions.npy", y_pred_ensemble)
print("✅ Ensemble predictions saved to ensemble_predictions.npy")

# ================================
# 🔹 8. Save Final Model Probabilities
# ================================
np.save("ensemble_probabilities.npy", final_proba)
print("✅ Ensemble probabilities saved to ensemble_probabilities.npy")

print("\n🎯 Pipeline completed successfully!")
