import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,accuracy_score
from sklearn.model_selection import train_test_split
from tested_model import OOFStackingClassifier

# ---------------- Page Config ----------------
st.set_page_config(layout="wide", page_title="Exoplanet Prediction Dashboard")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align: center;'>🪐 Extra Terrestrial Planet Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------------- Cached Loaders ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('Combined_new.csv')
    X = df.drop(columns=['label','StarID','Name'])
    y = df['label']
    return df,X, y

@st.cache_resource
def load_model():
    with open('Stacked.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

df,X, y = load_data()
model = load_model()

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("🔭 Prediction", "📊 Model Evaluation", "⚙️ Model Pipeline")
)

# ---------------- Shared Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y
)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
# ==========================================================
# PREDICTION PAGE
# ==========================================================
if page == "🔭 Prediction":
    st.subheader("Predict Planet Status by Star ID")

    star_id = st.text_input("Enter Star ID:")
    predict_btn = st.button("🔍 Predict Planetary Status")

    if predict_btn:
        if star_id:
            try:
                # Try to convert to int if StarID is numeric
                try:
                    star_id = int(star_id)
                except ValueError:
                    pass  # keep as string if not numeric

                # Get all rows (planets) for this StarID from the full dataframe
                star_rows = df[df['StarID'] == star_id]

                if star_rows.empty:
                    st.error("⚠️ Star ID not found in the dataset.")
                else:
                    # Use the same feature structure as during training
                    star_features = X.loc[star_rows.index]

                    # Predict for all planets belonging to this star
                    preds = model.predict(star_features)
                    probs = model.predict_proba(star_features)[:, 1]

                    # Find likely planets
                    planet_mask = preds == 1
                    detected_planets = star_rows.loc[planet_mask, 'PlanetID'].tolist()
                    detected_probs = probs[planet_mask]

                    if len(detected_planets) > 0:
                        st.success(f"🌍 Total detected planets for StarID '{star_id}': **{len(detected_planets)}**")

                        # Create a clean results table
                        results_df = pd.DataFrame({
                            "PlanetID": detected_planets,
                            "Prediction_Probability": detected_probs
                        }).reset_index(drop=True)

                        st.dataframe(results_df.style.format({
                            "Prediction_Probability": "{:.2f}"
                        }))
                    else:
                        st.warning(f"❓ No planets detected for StarID '{star_id}' — status: **Unknown**")

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
        else:
            st.warning("Please enter a Star ID before predicting.")

# ==========================================================
# MODEL EVALUATION PAGE
# ==========================================================
elif page == "📊 Model Evaluation":
    st.subheader("Model Evaluation Metrics and Visualization")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test,y_pred)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%")
    with col2:
        st.pyplot(fig)

    

    

# ==========================================================
# ⚙️ MODEL PIPELINE PAGE
# ==========================================================
elif page == "⚙️ Model Pipeline":
    st.subheader("Model Pipeline Structure")

    st.write("Below is the model pipeline and structure used in this project:")
    st.code(str(model), language='python')

    st.markdown("""
    ### Notes:
    - The model uses an **Out-Of-Fold Stacking Classifier**.
    - It combines predictions from multiple base learners and a meta-learner for improved performance.
    - The stacking strategy helps reduce overfitting while boosting accuracy.
    """)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("<hr><p style='text-align:center;'>© 2025 Exoplanet Predictor | Built with Streamlit </p>", unsafe_allow_html=True)
