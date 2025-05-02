import re
import streamlit as st
from tasks.cleaning import clean_data
from tasks.eda import generate_auto_eda
from tasks.train import train_model
from llm_router import route_query

def extract_target_column(user_query, df_columns):
    user_query = user_query.lower().replace("_", " ")

    patterns = [
        r"predict ([\w\s]+)",
        r"target ([\w\s]+)",
        r"on ([\w\s]+)",
        r"train model on ([\w\s]+)",
        r"for ([\w\s]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, user_query)
        if match:
            target_candidate = match.group(1).strip().lower()
            for col in df_columns:
                cleaned_col = col.lower().replace("_", " ").strip()
                if target_candidate in cleaned_col or cleaned_col in target_candidate:
                    return col
    return None

def handle_query(user_query, df):
    if df is None:
        st.error("❌ No dataset found. Please upload a CSV first.")
        return None

    agent = route_query(user_query)

    if agent == "unknown" and any(kw in user_query.lower() for kw in ["predict", "train", "model"]):
        agent = "train"

    if agent == "cleaning":
        return clean_data(df)
    elif agent == "eda":
        return generate_auto_eda(df)
    elif agent == "train":
        target = extract_target_column(user_query, df.columns)
        if target:
            return train_model(df, target)
        else:
            st.error("❌ Could not detect target column. Please mention it clearly in your query.")
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