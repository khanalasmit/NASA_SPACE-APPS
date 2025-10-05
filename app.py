import streamlit as st
import pickle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from tested_model import OOFStackingClassifier

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>A Extra Terrestrial Planet Prediction Model</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    Kepler_data = pd.read_csv('Training_data.csv')
    X = Kepler_data.drop(columns=['koi_disposition'])
    y = Kepler_data['koi_disposition']
    return X, y

@st.cache_resource
def load_model():
    with open('Stacked.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load once
X, y = load_data()
stack_clf = load_model()

# Train/test split (fast enough)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict
y_pred_proba = stack_clf.predict_proba(X_test)[:, 1]
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

st.pyplot(fig)
