import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from tasks.cleaning import clean_data
import numpy as np

def detect_task_type(y, target_name):
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 10:
            return "classification"
        else:
            return "regression"
    elif y.dtype == 'object' or y.dtype.name == 'category':
        return "classification"
    elif any(keyword in target_name.lower() for keyword in ["price", "score", "income", "age"]):
        return "regression"
    return "classification"

def train_model(df, target):
    st.header("🧠 Hardcoded Model Training Agent")

    if target not in df.columns:
        st.error(f"❌ Target column '{target}' not found in dataset.")
        return df, None, None

    # Step 1: Clean the dataset using your own agent
    st.info("🧼 Cleaning dataset before training...")
    df = clean_data(df)
    st.success("✅ Dataset cleaned.")

    # Step 2: Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Use improved task detection
    task_type = detect_task_type(y, target)
    st.info(f"📊 Detected task type: **{task_type.capitalize()}**")

    results = []

    # Step 4: Train Models
    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest Classifier": RandomForestClassifier()
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred, average='macro'), 4)
            })
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor()
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "R2 Score": round(r2_score(y_test, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            })

    # Step 5: Show results
    results_df = pd.DataFrame(results)
    st.subheader("📈 Model Evaluation")
    st.dataframe(results_df)

    return df, results_df, task_type
