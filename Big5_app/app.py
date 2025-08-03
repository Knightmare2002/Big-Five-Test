import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ================================
# 🔹 Load models
# ================================
st.set_page_config(page_title="Big Five Personality Test", page_icon="🧠", layout="centered")
st.title("🧠 Big Five Personality Test (ML-powered)")

@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_optuna.pkl")
    mlp = load_model("mlp_tuning.h5")
    return xgb, mlp

xgb_model, mlp_model = load_models()

# ================================
# 🔹 Big Five questions grouped by category
# ================================
questions = {
    "Extraversion (E)": [
        "I am the life of the party.",
        "I don't talk a lot.",
        "I feel comfortable around people.",
        "I keep in the background.",
        "I start conversations.",
        "I have little to say.",
        "I talk to a lot of different people at parties.",
        "I don't like to draw attention to myself.",
        "I don't mind being the center of attention.",
        "I am quiet around strangers."
    ],
    "Neuroticism (N)": [
        "I get stressed out easily.",
        "I am relaxed most of the time.",
        "I worry about things.",
        "I seldom feel blue.",
        "I am easily disturbed.",
        "I get upset easily.",
        "I change my mood a lot.",
        "I have frequent mood swings.",
        "I get irritated easily.",
        "I often feel blue."
    ],
    "Agreeableness (A)": [
        "I feel little concern for others.",
        "I am interested in people.",
        "I insult people.",
        "I sympathize with others' feelings.",
        "I am not interested in other people's problems.",
        "I have a soft heart.",
        "I am not really interested in others.",
        "I take time out for others.",
        "I feel others' emotions.",
        "I make people feel at ease."
    ],
    "Conscientiousness (C)": [
        "I am always prepared.",
        "I leave my belongings around.",
        "I pay attention to details.",
        "I make a mess of things.",
        "I get chores done right away.",
        "I often forget to put things back in their proper place.",
        "I like order.",
        "I shirk my duties.",
        "I follow a schedule.",
        "I am exacting in my work."
    ],
    "Openness (O)": [
        "I have a rich vocabulary.",
        "I have difficulty understanding abstract ideas.",
        "I have a vivid imagination.",
        "I am not interested in abstract ideas.",
        "I have excellent ideas.",
        "I do not have a good imagination.",
        "I am quick to understand things.",
        "I use difficult words.",
        "I spend time reflecting on things.",
        "I am full of ideas."
    ]
}

# ================================
# 🔹 Interactive questionnaire
# ================================
st.markdown("### 📋 Please answer the 50 questions (1 = Strongly Disagree, 5 = Strongly Agree)")

responses = []
for category, qs in questions.items():
    st.subheader(category)
    for q in qs:
        score = st.selectbox(f"{q}", options=[1, 2, 3, 4, 5], index=2, key=f"{category}_{q}")
        responses.append(score)

# ================================
# 🔹 Radar chart function
# ================================
def plot_radar(ocean_scores):
    labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    scores = ocean_scores.tolist() + ocean_scores.tolist()[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_yticks([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("OCEAN Profile Radar Chart")
    st.pyplot(fig)

# ================================
# 🔹 Prediction
# ================================
if st.button("🚀 Get Your Personality Profile"):
    X_input = np.array(responses, dtype=float).reshape(1, -1)

    # Model predictions
    prob_xgb = xgb_model.predict_proba(X_input)
    prob_mlp = mlp_model.predict(X_input)

    # Soft Voting Ensemble
    w_xgb, w_mlp = 0.3, 0.7
    final_prob = w_xgb * prob_xgb + w_mlp * prob_mlp
    pred_cluster = np.argmax(final_prob, axis=1)[0]

    # Cluster labels
    cluster_names = {0: "Reserved", 1: "Striver", 2: "Internalizer", 3: "Balanced"}

    # Calculate mean OCEAN scores (for radar chart)
    O = np.mean(responses[40:50])
    C = np.mean(responses[30:40])
    E = np.mean(responses[0:10])
    A = np.mean(responses[20:30])
    N = np.mean(responses[10:20])
    ocean_scores = np.array([O, C, E, A, N]) / 5  # normalize 0-1 for radar

    # Display results
    st.subheader("🎯 Your Predicted Profile")
    st.success(f"**{cluster_names[pred_cluster]}**")

    # === Bar chart ===
    st.markdown("### 🔹 Class Probabilities")
    cluster_names = {0: "Reserved", 1: "Striver", 2: "Internalizer", 3: "Balanced"}
    prob_df = pd.DataFrame([final_prob[0]], columns=[cluster_names[i] for i in range(final_prob.shape[1])])
    st.bar_chart(prob_df.T)  # Transpose for better visualization

    st.markdown("### 📊 Your OCEAN Mean Scores")
    st.write({
        "Openness": round(O, 2),
        "Conscientiousness": round(C, 2),
        "Extraversion": round(E, 2),
        "Agreeableness": round(A, 2),
        "Neuroticism": round(N, 2)
    })

    st.markdown("### 📊 Your OCEAN Trait Distribution")
    plot_radar(ocean_scores)
