import re
import streamlit as st
from tasks.cleaning import clean_data
from tasks.eda import generate_auto_eda
from tasks.train import train_model
from llm_router import route_query

def extract_target_column(user_query, df_columns):
    user_query = user_query.lower().replace("_", " ")
    patterns = [
        r"predict ([\w\s]+)", r"target ([\w\s]+)",
        r"on ([\w\s]+)", r"train model on ([\w\s]+)",
        r"for ([\w\s]+)", r"eda on ([\w\s]+)", r"explore ([\w\s]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, user_query)
        if match:
            candidate = match.group(1).strip().lower()
            for col in df_columns:
                if candidate in col.lower().replace("_", " ").strip():
                    return col
    return None

def infer_best_target(df):
    # Heuristic: target with high cardinality but < 80% unique (not ID column)
    for col in df.columns[::-1]:  # Try from the end (often target is last)
        if df[col].dtype in ["int64", "float64", "object"] and 2 <= df[col].nunique() < df.shape[0] * 0.8:
            return col
    return df.columns[-1]  # Fallback to last column

def handle_query(user_query, df):
    if df is None:
        st.error("❌ No dataset found.")
        return None

    agent = route_query(user_query)

    if agent == "unknown" and any(kw in user_query.lower() for kw in ["predict", "train", "model"]):
        agent = "train"

    if agent == "cleaning":
        return clean_data(df)
    elif agent == "eda":
        target = extract_target_column(user_query, df.columns)
        if not target:
            target = infer_best_target(df)
            st.warning(f"⚠️ Target column not mentioned. Using inferred target: `{target}`")
        return generate_auto_eda(df, target)
    elif agent == "train":
        target = extract_target_column(user_query, df.columns)
        if target:
            return train_model(df, target)
        else:
            st.error("❌ Could not detect target column.")
            return df
    elif agent == "compare":
        if 'results_df' in st.session_state and st.session_state.results_df is not None:
            from tasks.compare_models import explain_and_plot
            explain_and_plot(st.session_state.results_df, st.session_state.task_type)
            return "✅ Showing model comparison."
        else:
            return "⚠️ No model results available to compare."
    else:
        return "🤔 I couldn't understand your query. Try asking to clean, explore, or train your dataset!"