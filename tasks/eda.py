import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import skew

def generate_auto_eda(df):
    st.header("📊 Exploratory Data Analysis (EA Agent)")

    # Target Selection
    target = st.selectbox("Select your Target Column for EDA:", df.columns)

    if target is None or target == "":
        st.warning("Please select a target column to proceed.")
        return

    # Basic Overview
    st.subheader("Dataset Overview")
    st.write(f"✅ Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.write(f"✅ Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}")
    st.write(f"✅ Categorical Features: {len(df.select_dtypes(include=['object']).columns)}")

    # Detect Task Type
    if df[target].dtype == 'object' or df[target].nunique() <= 10:
        task_type = "classification"
    else:
        task_type = "regression"

    st.info(f"Detected Task Type: **{task_type.capitalize()}**")

    # Missing Values
    st.subheader("Missing Value Report")
    missing_percent = df.isnull().mean() * 100
    missing_report = missing_percent[missing_percent > 0].sort_values(ascending=False)
    if not missing_report.empty:
        st.dataframe(missing_report.to_frame("Missing %"))
    else:
        st.success("✅ No missing values detected.")

    # If Classification
    if task_type == "classification":
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=target, data=df, ax=ax)
        ax.set_title("Target Class Distribution")
        st.pyplot(fig)

        st.subheader("Feature vs Target Plots (Numeric Features)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != target:
                fig, ax = plt.subplots()
                sns.boxplot(x=target, y=col, data=df, ax=ax)
                ax.set_title(f"{col} by {target}")
                st.pyplot(fig)

        st.subheader("Feature vs Target Plots (Categorical Features)")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target:
                fig, ax = plt.subplots()
                sns.countplot(x=col, hue=target, data=df, ax=ax)
                ax.set_title(f"{col} distribution by {target}")
                st.pyplot(fig)

    # If Regression
    else:
        st.subheader("Skewness and Distribution Plots")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewness_report = []
        for col in numeric_cols:
            if col != target:
                col_skew = skew(df[col].dropna())
                skewness_report.append((col, round(col_skew, 2)))
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col} (Skewness: {round(col_skew,2)})")
                st.pyplot(fig)

        skewness_df = pd.DataFrame(skewness_report, columns=["Feature", "Skewness"])
        st.dataframe(skewness_df)

        st.subheader("Correlation Heatmap")
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric features for correlation analysis.")

    # Recommendations
    st.subheader("Recommendations")

    if task_type == "classification":
        class_counts = df[target].value_counts(normalize=True)
        if class_counts.max() > 0.8:
            st.warning("⚠️ Severe class imbalance detected. Consider resampling techniques.")
    else:
        high_skew = skewness_df[(skewness_df['Skewness'] > 1) | (skewness_df['Skewness'] < -1)]
        if not high_skew.empty:
            for feature in high_skew['Feature']:
                st.warning(f"⚠️ Feature '{feature}' is highly skewed — consider log/cube root transformation.")

    if not missing_report.empty:
        high_missing = missing_report[missing_report > 50]
        if not high_missing.empty:
            for feature in high_missing.index:
                st.warning(f"⚠️ Feature '{feature}' has over 50% missing — consider dropping or careful imputation.")

    st.success("✅ EA Agent analysis completed.")