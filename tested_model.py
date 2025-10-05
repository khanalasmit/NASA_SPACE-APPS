# ...existing code...
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pickle

Kepler_data=pd.read_csv('Combined.csv')
Kepler_data.info()
X=Kepler_data.drop(columns=['label'])
y=Kepler_data['label']
le=LabelEncoder()
le.fit(y)
y_trf=le.transform(y)
print(le.classes_)
X_train,X_test,y_train,y_test=train_test_split(X,y_trf,test_size=0.2,random_state=42,stratify=y)

class OOFStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Out-Of-Fold stacking classifier that builds default base learners and a default meta-learner
    if none are provided. Use base_models/meta_model kwargs to override.
    """
    def __init__(self, base_models=None, meta_model=None, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

        # default base learners
        if base_models is None:
            rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=None,
                                        max_samples=0.8, min_samples_leaf=1, n_estimators=500,
                                        oob_score=True, random_state=self.random_state)
            xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                subsample=1, colsample_bytree=1,
                                eval_metric='logloss', use_label_encoder=False, random_state=self.random_state)
            gb = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=500,
                                            subsample=1, random_state=self.random_state)
            lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=self.random_state)
            svc = SVC(C=2.0, kernel='rbf', probability=True, random_state=self.random_state)
            base_models = [rf, xgb, gb, lgb, svc]

        # default meta learner
        if meta_model is None:
            meta_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=self.random_state)

        self.base_models = base_models
        self.meta_model = meta_model
        self.fitted_base_models = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((n_samples, len(self.base_models)))

        # Out-of-fold predictions for each base model
        for i, model in enumerate(self.base_models):
            oof = np.zeros(n_samples)
            for train_idx, val_idx in skf.split(X, y):
                mdl_clone = clone(model)
                mdl_clone.fit(X[train_idx], y[train_idx])
                # try predict_proba, fallback to decision_function, fallback to predict
                if hasattr(mdl_clone, "predict_proba"):
                    oof[val_idx] = mdl_clone.predict_proba(X[val_idx])[:, 1]
                elif hasattr(mdl_clone, "decision_function"):
                    # scale decision_function to [0,1] via sigmoid-like mapping
                    df = mdl_clone.decision_function(X[val_idx])
                    oof[val_idx] = 1 / (1 + np.exp(-df))
                else:
                    oof[val_idx] = mdl_clone.predict(X[val_idx])
            oof_preds[:, i] = oof

        # Train meta-model on OOF predictions
        self.meta_model.fit(oof_preds, y)

        # Retrain base models on full dataset and save them
        self.fitted_base_models = [clone(m).fit(X, y) for m in self.base_models]

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        # build meta-features from fully trained base models
        meta_features = np.column_stack([
            (m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba")
             else (1 / (1 + np.exp(-m.decision_function(X)))) if hasattr(m, "decision_function")
             else m.predict(X))
            for m in self.fitted_base_models
        ])
        # return meta-model probabilities if available, else wrap single-column scores
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        else:
            probs = self.meta_model.predict(meta_features)
            # ensure shape (n_samples, 2)
            probs = np.vstack([1 - probs, probs]).T
            return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        # assume binary prob in column 1
        return (probs[:, 1] > 0.5).astype(int)

        

# Example usage (keeps previous notebook variables/flow)
# ...existing code...
stack_clf = OOFStackingClassifier(n_splits=5, random_state=42)   # uses defaults defined above
stack_clf.fit(X_train, y_train)
y_pred = stack_clf.predict(X_test)
y_pred_proba = stack_clf.predict_proba(X_test)[:, 1]
stack_clf.feature_importances_()
# write using a file object
with open(r'Stacked.pkl', 'wb') as f:
    pickle.dump(stack_clf, f)
# ...existing code...