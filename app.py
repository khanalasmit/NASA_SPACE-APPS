import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.ensemble import BaggingClassifier,StackingClassifier,RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier as XGC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC


Kepler_data=pd.read_csv('Training_data.csv')
X=Kepler_data.drop(columns=['koi_disposition'])
y=Kepler_data['koi_disposition']
le=LabelEncoder()
le.fit(y)
y_trf=le.transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y_trf,test_size=0.2,random_state=42,stratify=y)

class OOFStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model, n_splits=5, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.random_state = random_state
        self.fitted_base_models = []  # will hold final trained clones of base models

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        # Out-of-fold predictions
        for i, model in enumerate(self.base_models):
            oof = np.zeros(n_samples)
            for train_idx, val_idx in skf.split(X, y):
                mdl_clone = clone(model)
                mdl_clone.fit(X[train_idx], y[train_idx])
                oof[val_idx] = mdl_clone.predict_proba(X[val_idx])[:, 1]
            oof_preds[:, i] = oof

        # Train meta-model on OOF predictions
        self.meta_model.fit(oof_preds, y)

        # Retrain base models on full dataset
        self.fitted_base_models = [clone(m).fit(X, y) for m in self.base_models]

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        # Get predictions from fully trained base models
        meta_features = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self.fitted_base_models
        ])
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# Define base learners
rf = RandomForestClassifier(bootstrap= True, criterion= 'gini', max_depth= None, 
                            max_samples= 0.8, min_samples_leaf= 1, n_estimators= 500, oob_score=True)

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=1, colsample_bytree=1,
                    eval_metric='logloss', use_label_encoder=False, random_state=42)

gb = GradientBoostingClassifier(learning_rate= 0.1, max_depth= 3, n_estimators= 500,
                                 subsample=1, random_state=42)


lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)

svc = SVC(C=2.0, kernel='rbf', probability=True, random_state=42)

base_models = [rf, xgb, gb, lgb, svc]

# Option 1: Logistic Regression as meta learner
meta_log = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=500, random_state=42)

# Option 2: Random Forest as meta learnern
meta_rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

# Build stacking model (just swap meta model here)
stack_clf = OOFStackingClassifier(base_models=base_models, meta_model=meta_rf, n_splits=5)

# Fit and predict
stack_clf.fit(X_train, y_train)
y_pred = stack_clf.predict(X_test)
y_pred_proba = stack_clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend()

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center;'>A Extra Teresterial Planet prediction model</h1>", unsafe_allow_html=True)
# Display in Streamlit
st.pyplot(fig)


