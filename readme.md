# Exoplanet Detection & B-Tree Visualization Project

## Project Overview

This repository contains a complete workflow for exoplanet detection using Kepler and TESS data, including feature engineering, model training and hyperparameter tuning, transfer learning, and a Streamlit-based B-tree visualization app.

---

## Folder Structure

```
.
├── app.py
├── requirements.txt
├── Combined_new.csv
├── Combined.csv
├── data_combining.ipynb
├── kepler_features.csv
├── kpoi_data.ipynb
├── model_testing.ipynb
├── readme.md
├── Stacked.pkl
├── tess_data.ipynb
├── tess_features.csv
├── tested_model.py
├── testing_tess.csv
├── Training_data.csv
├── unused/
│   ├── Candidates.csv
│   ├── data_scaling.ipynb
│   ├── koi_toi_combined_features.csv
│   ├── tested_model.ipynb
│   └── tested_model_transfer.ipynb
└── data/
		├── cumulative_2025.10.01_04.13.28.csv
		└── TOI_2025.10.01_04.13.59.csv
```

---

## Workflow

### 1. **Feature Engineering**

- `tess_data.ipynb`:  
	Processes raw TESS data, maps columns to Kepler-like features, and saves as `testing_tess.csv`.

- `kpoi_data.ipynb`:  
	Processes Kepler cumulative data, selects safe features, encodes dispositions, and saves as `Training_data.csv`.

- `data_combining.ipynb`:  
	Combines engineered Kepler and TESS features, performs cleaning, and outputs `Combined_new.csv`.

### 2. **Model Training & Hyperparameter Tuning**

- `model_testing.ipynb`:  
	Loads combined features, splits data, performs model selection (Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM), and hyperparameter tuning using `GridSearchCV`. Implements stacking with out-of-fold predictions.

### 3. **Model Module**

- `tested_model.py`:  
	Contains the final model class and code for training, prediction, and stacking logic. This module can be imported for inference or further development.

### 4. **Streamlit App**

- `app.py`:  
	A Streamlit application for B-tree visualization. Allows users to interactively insert and delete nodes in a B-tree and see the updated structure as an image.

---

## Requirements

All dependencies are listed in `requirements.txt`.  
Key packages include:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- streamlit
- Pillow

Install with:
```sh
pip install -r requirements.txt
```

---

## Usage

### **Feature Engineering & Model Training**
1. Run the notebooks in order:
	 - `tess_data.ipynb`
	 - `kpoi_data.ipynb`
	 - `data_combining.ipynb`
	 - `model_testing.ipynb`

2. Use `tested_model.py` for model inference or further training.

### **B-Tree Visualization App**
1. Ensure all requirements are installed.
2. Run the Streamlit app:
	 ```sh
	 streamlit run app.py
	 ```
3. Use the sidebar to insert or delete nodes and visualize the B-tree.

---

## Notes

- The `unused/` folder contains legacy or experimental notebooks and models.
- Data files are expected in the root or `data/` directory as referenced in the notebooks.
- For transfer learning and advanced analysis, see the notebooks in `unused/`.

---

## Authors

- [Asmit Khanal\
   Asmit Kumar khanal]

---

