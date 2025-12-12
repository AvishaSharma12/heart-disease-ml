❤️ Heart Disease Prediction – Machine Learning Project
👩‍💻 Developed by: Avisha Sharma
This project uses machine learning models to predict the likelihood of heart disease using clinical patient data.It includes data preprocessing, model training, evaluation, hyperparameter tuning and model saving for deployment.
📌 Project Overview
This repository contains:
Data preprocessing
Feature selection
Training six different ML models
Hyperparameter tuning using GridSearchCV
Performance evaluation (Accuracy, F1, ROC-AUC)
Saving trained models using Joblib
Files ready for deployment or integration into apps
| Model                  | Status            | Approx Accuracy/ROC-AUC |
| ---------------------- | ----------------- | ----------------------- |
| Logistic Regression    | ✔ Best performing | ~90%                    |
| Random Forest          | ✔ Tuned           | ~85–92%                 |
| K-Nearest Neighbors    | ✔                 | ~64%                    |
| Support Vector Machine | ✔                 | ~70%                    |
| Decision Tree          | ✔                 | ~80%                    |
| Gradient Boosting      | ✔                 | ~79%                    |
Final Best Models: Logistic Regression or Random Forest
| File                      | Description                           |
| ------------------------- | ------------------------------------- |
| `heart_disease_ml.ipynb`  | Full Jupyter/Colab notebook with code |
| `best_model.pkl`          | Best performing ML model              |
| `final_model.pkl`         | Final saved model                     |
| `random_forest_model.pkl` | Random Forest model                   |
| `scaler_heart.pkl`        | StandardScaler used in training       |
1️⃣ Install requirements
```bash
pip install joblib scikit-learn pandas numpy
2️⃣ Load the model & scaler
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler_heart.pkl")
3️⃣ Prepare sample input
new_data = pd.DataFrame({
    "age": [63],
    "sex": [1],
    "cp": [3],
    "trestbps": [145],
    "chol": [233],
    "fbs": [1],
    "restecg": [0],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [2.3],
    "slope": [0],
    "ca": [0],
    "thal": [1]
})
4️⃣ Scale & predict
scaled = scaler.transform(new_data)
prediction = model.predict(scaled)

print("Prediction:", prediction)
