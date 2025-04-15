import streamlit as st
import pandas as pd
from agents import executor_agent, responder_agent
from utils import save_dataframe_as_csv
from semantic_router import extract_relevant_columns
from intent_router.router import classify_user_intent

st.set_page_config(page_title="Agentic AI v5 - Semantic + Mistral", layout="wide")
st.title("🧠 Agentic AI v5 - Semantic Column Detection + Mistral Reasoning")

if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False

file = st.file_uploader("📂 Upload your dataset (CSV/XLSX):", type=["csv", "xlsx"])
if file:
    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    st.session_state.uploaded_df = df
    st.session_state.upload_complete = True
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())

if st.session_state.upload_complete:
    for msg in st.session_state.chat_memory:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything like 'Clean data', 'Train model', 'Plot histogram of sales', etc.")
    if user_input:
        st.session_state.chat_memory.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):

            cols = extract_relevant_columns(user_input, st.session_state.uploaded_df.columns.tolist())
            intent_result = classify_user_intent(user_input, cols)

            intent = intent_result["intents"][0]
            rewritten_query = intent_result["rewritten_query"]
            explanation = intent_result["explanation"]

            response, updated_df, output_type = executor_agent(intent, rewritten_query, st.session_state.uploaded_df)
            st.session_state.uploaded_df = updated_df
            display_output = responder_agent(intent, response, output_type)

            result_to_show = f"**Task Summary:** {explanation}\n\n{display_output}"
            st.markdown(result_to_show)

            st.dataframe(st.session_state.uploaded_df.head())
            save_dataframe_as_csv(st.session_state.uploaded_df)
            st.download_button("📥 Download Updated Dataset", data=open("updated_dataset.csv", "rb"),
                               file_name="updated_dataset.csv", mime="text/csv")

            st.session_state.chat_memory.append({"role": "assistant", "content": result_to_show})
else:
    st.info("👋 Please upload your dataset to get started.")