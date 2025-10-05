import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tested_model import OOFStackingClassifier

# ---------------- Page Config ----------------
st.set_page_config(layout="wide", page_title="Exoplanet Prediction Dashboard")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align: center;'>🪐 Extra Terrestrial Planet Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------------- Cached Loaders ----------------
@st.cache_data
def load_data():
    df = pd.read_csv('Training_data.csv')
    X = df.drop(columns=['koi_disposition'])
    y = df['koi_disposition']
    return X, y

@st.cache_resource
def load_model():
    with open('Stacked.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

X, y = load_data()
model = load_model()

# ---------------- Sidebar Navigation ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("🔭 Prediction", "📊 Model Evaluation", "⚙️ Model Pipeline")
)

# ---------------- Shared Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 🔭 PREDICTION PAGE
# ==========================================================
if page == "🔭 Prediction":
    st.subheader("Predict Planet Status by Star ID")

    star_id = st.text_input("Enter Star ID (index from dataset):")

    predict_btn = st.button("🔍 Predict Planetary Status")

    if predict_btn:
        if star_id:
            try:
                star_id = int(star_id)
                star_features = X.iloc[star_id].values.reshape(1, -1)

                pred_prob = model.predict_proba(star_features)[0, 1]
                pred_class = model.predict(star_features)[0]

                st.success(f"Prediction Probability (Planet Likely): **{pred_prob:.2f}**")
                st.info(f"Predicted Class: **{'Planet' if pred_class == 1 else 'Not Planet'}**")

            except ValueError:
                st.error("⚠️ Please enter a valid numeric Star ID.")
            except IndexError:
                st.error("⚠️ Star ID out of range. Please enter a valid ID.")
        else:
            st.warning("Please enter a Star ID before predicting.")

# ==========================================================
# 📊 MODEL EVALUATION PAGE
# ==========================================================
elif page == "📊 Model Evaluation":
    st.subheader("Model Evaluation Metrics and Visualization")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = (y_pred == y_test).mean()

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

    # Feature Importance (if available)
    st.markdown("### Feature Importance")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = np.array(X.columns)[sorted_idx]

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(sorted_features[:15][::-1], importances[sorted_idx][:15][::-1])
        ax2.set_xlabel("Importance")
        ax2.set_title("Top 15 Feature Importances")
        st.pyplot(fig2)
    else:
        st.info("Feature importances are not available for this model.")

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
