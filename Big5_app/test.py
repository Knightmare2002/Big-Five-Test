import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ================================
# 🔹 Load models
# ================================
st.title("🧠 Big Five Personality Test with ML Prediction")

@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_optuna.pkl")
    mlp = load_model("mlp_tuning.keras")
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
# 🔹 Manual input mode
# ================================
st.subheader("🛠️ Manual Input Mode (Optional)")
manual_input = st.text_area("Paste 50 values separated by commas (1-5)", "")

use_manual = False
responses = []

if manual_input:
    try:
        responses = [float(x.strip()) for x in manual_input.split(",")]
        if len(responses) == 50:
            use_manual = True
            st.success("✅ Manual input accepted.")
        else:
            st.error("❌ You must provide exactly 50 values.")
    except ValueError:
        st.error("❌ Invalid format. Use numbers separated by commas.")

# ================================
# 🔹 Interactive questionnaire (if not using manual input)
# ================================
if not use_manual:
    st.header("📋 Answer the 50 questions (1 = Strongly Disagree, 5 = Strongly Agree)")

    for category, qs in questions.items():
        st.subheader(category)
        for i, q in enumerate(qs, 1):
            score = st.selectbox(f"{q}", options=[1, 2, 3, 4, 5], index=2, key=f"{category}_{i}")
            responses.append(score)

# ================================
# 🔹 Debug input
# ================================
st.markdown("### 🛠️ Debug: Collected Responses")
st.write(responses)

# ================================
# 🔹 Prediction
# ================================
if st.button("🔍 Predict Personality"):
    X_input = np.array(responses, dtype=float).reshape(1, -1)
    st.write("📊 **Input shape sent to model:**", X_input.shape)
    st.write("📊 **Input values:**", X_input.tolist())

    # Model predictions
    prob_xgb = xgb_model.predict_proba(X_input)
    prob_mlp = mlp_model.predict(X_input)

    # Soft Voting Ensemble
    w_xgb, w_mlp = 0.3, 0.7
    final_prob = w_xgb * prob_xgb + w_mlp * prob_mlp
    pred_cluster = np.argmax(final_prob, axis=1)[0]

    # Cluster names
    cluster_names = {0: "Reserved", 1: "Striver", 2: "Internalizer", 3: "Balanced"}

    # Output prediction
    st.subheader("🎯 Predicted Profile")
    st.success(f"**{cluster_names[pred_cluster]}**")
    st.write("🔹 Class probabilities:", final_prob.tolist())
