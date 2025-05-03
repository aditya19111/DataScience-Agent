import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import skew
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def run_target_eda(df, target):
    st.header("📊 Exploratory Data Analysis")

    if target not in df.columns:
        st.error(f"❌ Target column '{target}' not found.")
        return

    task_type = "classification" if df[target].dtype == "object" or df[target].nunique() <= 10 else "regression"
    st.info(f"Detected task type: {task_type.title()}")

    st.subheader("📌 Dataset Overview")
    st.write(f"✅ Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Missing values
    st.subheader("🧱 Missing Values")
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]
    st.dataframe(missing.to_frame("Missing %") if not missing.empty else pd.DataFrame(columns=["Missing %"]))

    # Skewness
    st.subheader("📉 Skewness & Distribution")
    skewness = {}
    for col in df.select_dtypes(include=np.number).columns:
        if col != target:
            sk = skew(df[col].dropna())
            skewness[col] = round(sk, 2)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"{col} (Skew: {sk:.2f})")
            st.pyplot(fig)
    skew_df = pd.DataFrame.from_dict(skewness, orient="index", columns=["Skewness"])
    st.dataframe(skew_df)

    # Correlation heatmap
    st.subheader("📊 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)
    if len(numeric_cols.columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Outliers
    st.subheader("🚨 Outlier Summary (IQR)")
    outliers = {}
    for col in numeric_cols.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers[col] = ((df[col] < lower) | (df[col] > upper)).sum()
    st.dataframe(pd.DataFrame.from_dict(outliers, orient="index", columns=["Outlier Count"]))

    # Feature importance
    st.subheader("⭐ Feature Importance")
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    try:
        if task_type == "classification":
            imp = mutual_info_classif(X.fillna(0), y)
        else:
            imp = mutual_info_regression(X.fillna(0), y)
        st.dataframe(pd.DataFrame({"Feature": X.columns, "Importance": imp}).sort_values(by="Importance", ascending=False))
    except:
        st.warning("⚠️ Feature importance could not be calculated.")

    # Target distribution
    st.subheader("📈 Target Distribution")
    fig, ax = plt.subplots()
    if task_type == "classification":
        sns.countplot(x=target, data=df, ax=ax)
    else:
        sns.histplot(df[target], kde=True, ax=ax)
    st.pyplot(fig)

    # Categorical visualizations
    if task_type == "classification":
        for col in df.select_dtypes(include="object"):
            if col != target:
                fig, ax = plt.subplots()
                sns.countplot(x=col, hue=target, data=df, ax=ax)
                ax.set_title(f"{col} by {target}")
                st.pyplot(fig)
    else:
        for col in df.select_dtypes(include="object"):
            fig, ax = plt.subplots()
            sns.boxplot(x=col, y=target, data=df, ax=ax)
            ax.set_title(f"{target} by {col}")
            st.pyplot(fig)

    # Recommendations
    st.subheader("🧠 Recommendations")
    for feature, val in skewness.items():
        if abs(val) > 1:
            st.warning(f"Feature '{feature}' is highly skewed (Skew: {val}). Consider transformation.")

    if not missing.empty:
        for col, pct in missing.items():
            if pct > 50:
                st.warning(f"Feature '{col}' has {pct:.2f}% missing — consider dropping or special imputation.")

    if task_type == "classification":
        dist = df[target].value_counts(normalize=True)
        if dist.max() > 0.8:
            st.warning("Severe class imbalance detected — consider oversampling or class weights.")

    # Model recommendations
    st.subheader("🤖 Model Recommendations")
    if task_type == "classification":
        if df[target].nunique() <= 2:
            st.markdown("- Logistic Regression (baseline)")
            st.markdown("- Random Forest Classifier")
        else:
            st.markdown("- Random Forest Classifier")
            st.markdown("- XGBoost or LightGBM for tabular features")
        if dist.max() > 0.8:
            st.markdown("- Consider Logistic Regression with `class_weight='balanced'`")
    else:
        st.markdown("- Linear Regression (if low skew and correlation exists)")
        st.markdown("- Random Forest Regressor (if outliers present)")
        if any(abs(val) > 1 for val in skewness.values()):
            st.markdown("- Consider log-transforming skewed features before using Linear Regression")
        st.markdown("- XGBoost Regressor for robust performance on wide/tabular data")

def generate_auto_eda(df, target):
    run_target_eda(df, target)