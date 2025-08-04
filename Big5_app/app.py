import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# ================================
# 🔹 Load models
# ================================
st.set_page_config(page_title="Big Five Personality Test", page_icon="🧠", layout="centered")
st.title("🧠 Big Five Personality Test (ML-powered)")

@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_optuna.pkl")
    mlp = load_model("mlp_tuning.keras")
    return xgb, mlp

xgb_model, mlp_model = load_models()

# ================================
# 🔹 Connect to Google Sheets (if online)
# ================================
def get_gsheet_client():
    try:
        #st.write("🔍 DEBUG: Trying to read gcp_service_account from secrets...") #DEBUG

        creds_dict = st.secrets["gcp_service_account"]

        #st.write("✅ DEBUG: Successfully read credentials keys:", list(creds_dict.keys())) #DEBUG

        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes=[
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ])

        client = gspread.authorize(creds)

        #st.write("✅ DEBUG: Google Sheets client authorized successfully") #DEBUG

        return client
    except Exception as e:
        st.error(f"❌ DEBUG: Failed to connect to Google Sheets: {e}")
        return None


# ================================
# 🔹 Big Five questions grouped by category
# ================================
questions = {
    "Extraversion (E)": [
        "I am the life of the party.", "I don't talk a lot.", "I feel comfortable around people.",
        "I keep in the background.", "I start conversations.", "I have little to say.",
        "I talk to a lot of different people at parties.", "I don't like to draw attention to myself.",
        "I don't mind being the center of attention.", "I am quiet around strangers."
    ],
    "Neuroticism (N)": [
        "I get stressed out easily.", "I am relaxed most of the time.", "I worry about things.",
        "I seldom feel blue.", "I am easily disturbed.", "I get upset easily.",
        "I change my mood a lot.", "I have frequent mood swings.", "I get irritated easily.",
        "I often feel blue."
    ],
    "Agreeableness (A)": [
        "I feel little concern for others.", "I am interested in people.", "I insult people.",
        "I sympathize with others' feelings.", "I am not interested in other people's problems.",
        "I have a soft heart.", "I am not really interested in others.", "I take time out for others.",
        "I feel others' emotions.", "I make people feel at ease."
    ],
    "Conscientiousness (C)": [
        "I am always prepared.", "I leave my belongings around.", "I pay attention to details.",
        "I make a mess of things.", "I get chores done right away.", "I often forget to put things back in their proper place.",
        "I like order.", "I shirk my duties.", "I follow a schedule.", "I am exacting in my work."
    ],
    "Openness (O)": [
        "I have a rich vocabulary.", "I have difficulty understanding abstract ideas.", "I have a vivid imagination.",
        "I am not interested in abstract ideas.", "I have excellent ideas.", "I do not have a good imagination.",
        "I am quick to understand things.", "I use difficult words.", "I spend time reflecting on things.",
        "I am full of ideas."
    ]
}

# Randomize order only once and store in session state
if "randomized_questions" not in st.session_state:
    question_list = [(trait, q) for trait, qs in questions.items() for q in qs]
    np.random.shuffle(question_list)
    st.session_state.randomized_questions = question_list
else:
    question_list = st.session_state.randomized_questions


# ================================
# 🔹 Descriptions for each predicted label
# ================================
label_descriptions = {
    "Reserved": "🟦 **Reserved** – You tend to be introspective and careful in your actions. You value your personal space and are not easily swayed by external chaos. You might be seen as calm, observant, and self-contained.",

    "Striver": "🔴 **Striver** – You aim high and push yourself to achieve more. With your high energy, ambition, and confidence, you often stand out in group settings. You thrive when facing challenges and pursuing your goals.",

    "Internalizer": "🟠 **Internalizer** – You may experience emotions deeply and often reflect inward. You are thoughtful, sensitive, and may prefer to work through things on your own. Despite challenges, this introspection can make you very self-aware.",

    "Balanced": "🟢 **Balanced** – You maintain emotional stability and handle life with a calm, adaptable approach. You’re neither overly introverted nor extroverted, and people likely see you as reliable and grounded."
}

# ================================
# 🔹 Disclaimer
# ================================
st.markdown("### 📋 Please answer the 50 questions (1 = Strongly Disagree, 5 = Strongly Agree)")
st.markdown("""
**Disclaimer:** This test is **not an objective psychological assessment**.  
It estimates the **most likely inclination** toward one of four categories:
**Reserved**, **Striver**, **Internalizer**, and **Balanced**.  
These categories were derived via **GMM clustering** and labeled manually based on cluster centroids.
The categorization was done using a soft voting based approach using two ML algorithms (XGB and MLP)
""")

# ================================
# 🔹 Interactive questionnaire
# ================================
responses_by_trait = {"E": [], "N": [], "A": [], "C": [], "O": []}
for i, (trait, q) in enumerate(question_list):
    score = st.selectbox(q, options=[1, 2, 3, 4, 5], index=2, key=f"Q{i}")
    trait_key = trait[-2] if "(" in trait else trait  # estrae 'E' da "Extraversion (E)"
    responses_by_trait[trait_key].append(score)


# Rebuild responses in correct order for the model
responses = responses_by_trait["E"] + responses_by_trait["N"] + \
            responses_by_trait["A"] + responses_by_trait["C"] + \
            responses_by_trait["O"]

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
# === Initialize variables to avoid NameError ===
if "pred_cluster" not in st.session_state:
    st.session_state.pred_cluster = None
if "pred_label" not in st.session_state:
    st.session_state.pred_label = None


if st.button("🚀 Get Your Personality Profile"):
    X_input = np.array(responses, dtype=float).reshape(1, -1)

    # Model predictions
    prob_xgb = xgb_model.predict_proba(X_input)
    prob_mlp = mlp_model.predict(X_input)

    # Soft Voting Ensemble
    w_xgb, w_mlp = 0.3, 0.7
    final_prob = w_xgb * prob_xgb + w_mlp * prob_mlp
    pred_cluster = np.argmax(final_prob, axis=1)[0]

    # Cluster labels mapping
    cluster_names = {0: "Reserved", 1: "Striver", 2: "Internalizer", 3: "Balanced"}
    pred_label = cluster_names[pred_cluster]

    st.session_state.pred_cluster = pred_cluster
    st.session_state.pred_label = pred_label

    # Compute OCEAN means for visualization
    O = np.mean(responses[40:50])
    C = np.mean(responses[30:40])
    E = np.mean(responses[0:10])
    A = np.mean(responses[20:30])
    N = np.mean(responses[10:20])
    ocean_scores = np.array([O, C, E, A, N]) / 5  # normalize for radar chart

    # === Display Results ===
    st.subheader("🎯 Your Predicted Profile")
    st.success(f"**{pred_label}**")
    st.markdown(label_descriptions[pred_label])  # ✅ Add description here

    # Probability bar chart with class names
    st.markdown("### 🔹 Class Probabilities")
    prob_df = pd.DataFrame([final_prob[0]], columns=[cluster_names[i] for i in range(final_prob.shape[1])])
    st.bar_chart(prob_df.T)

    # OCEAN Mean Scores
    st.markdown("### 📊 Your OCEAN Mean Scores")
    st.write({
        "Openness": round(O, 2),
        "Conscientiousness": round(C, 2),
        "Extraversion": round(E, 2),
        "Agreeableness": round(A, 2),
        "Neuroticism": round(N, 2)
    })

    # Radar Chart
    st.markdown("### 📊 Your OCEAN Trait Distribution")
    plot_radar(ocean_scores)


# === Optional demographic fields ===
st.markdown("### 🧾 Optional Demographic Information")
race = st.selectbox("Race (**1=Mixed Race**, **2=Arctic** (Siberian, Eskimo), **3=Caucasian** (European), **4=Caucasian** (Indian), **5=Caucasian** (Middle East), **6=Caucasian** (North African, Other), **7=Indigenous Australian**, **8=Native American**, **9=North East Asian** (Mongol, Tibetan, Korean Japanese, etc), **10=Pacific** (Polynesian, Micronesian, etc), **11=South East Asian** (Chinese, Thai, Malay, Filipino, etc), **12=West African, Bushmen, Ethiopian**, **13=Other** (**0=missed**))",[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=0)
age = st.number_input("Age", min_value=14, max_value=100, value=30)
engnat = st.selectbox("English Native (0= missed, 1=yes, 2=no)", [0, 1, 2], index=0)
gender = st.selectbox("Gender (0=Unknown, 1=male, 2=female, 3=other", [0, 1, 2, 3], index=0)
hand = st.selectbox("Hand (0=missed, 1=right, 2=left, 3=both", [0, 1, 2, 3], index=0)
country = st.text_input("Country", value="IT")

# ================================
# 🔹 Save to Google Sheets or Local CSV (with header if empty)
# ================================
if st.button("💾 Save Your Results", key="save_button"):
    if st.session_state.pred_cluster is None:
        st.error("⚠️ Please generate your profile first!")
    else:
        index_val = np.random.randint(100000, 999999)
        meta = [index_val, race, age, engnat, gender, hand, "webapp", country]
        row = meta + responses + [st.session_state.pred_cluster, st.session_state.pred_label]

        columns = ["index","race","age","engnat","gender","hand","source","country"] + \
                  [f"E{i}" for i in range(1,11)] + [f"N{i}" for i in range(1,11)] + \
                  [f"A{i}" for i in range(1,11)] + [f"C{i}" for i in range(1,11)] + \
                  [f"O{i}" for i in range(1,11)] + ["Cluster","Psych_Label"]

        df_entry = pd.DataFrame([row], columns=columns)

        # ✅ Try Google Sheets first
        client = get_gsheet_client()
        if client:
            try:
                
                #st.write("✅ DEBUG: Google Sheets client created")

                sheet_id = st.secrets.get("GSHEET_ID", "NOT FOUND")
                #st.write("✅ DEBUG: GSHEET_ID =", sheet_id)

                sheet = client.open_by_key(sheet_id).sheet1
                #st.write("✅ DEBUG: Successfully opened sheet")

                current_values = sheet.get_all_values()
                #st.write(f"✅ DEBUG: Current sheet rows count: {len(current_values)}")

                if len(current_values) == 0:
                    #st.write("✅ DEBUG: Sheet is empty, adding header...")
                    sheet.append_row([str(c) for c in columns])
                    #st.write("✅ DEBUG: Header added successfully")

                #st.write("✅ DEBUG: Now adding user row:", row)
                clean_row = [str(x) for x in row]
                clean_row = clean_row[:len(columns)]
                sheet.append_row(clean_row, value_input_option="RAW")
                st.success("✅ Data collected successfully")

                new_values = sheet.get_all_values()
                #st.write("✅ DEBUG: Sheet now has rows:", len(new_values))
                #st.write("✅ DEBUG: Last row in sheet:", new_values[-1])


            except Exception as e:
                st.error(f"❌ Google Sheets error: {e}")
        else:
            # ✅ Local CSV
            save_path = "user_big5_responses.csv"
            write_header = not os.path.exists(save_path) or os.path.getsize(save_path) == 0
            df_entry.to_csv(save_path, mode='a', header=write_header, index=False)
            st.success("✅ Saved locally (CSV with headers if new)!")
