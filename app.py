import streamlit as st
import pandas as pd
from query_manager import handle_query
from session_manager import save_df
from memory_manager import update_memory

if 'df' not in st.session_state:
    st.session_state.df = None
if 'memory' not in st.session_state:
    st.session_state.memory = []
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None

st.title("🤖 canny-Data Science Chatbot")
st.write("Upload your CSV and interact with the AI agent.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.success("✅ File uploaded successfully.")

if st.session_state.df is not None:
    st.subheader("Current Data Snapshot")
    st.dataframe(st.session_state.df.head())

    query = st.text_input("Ask your query (e.g., 'clean data', 'EDA', 'predict age')")

    if st.button("Submit"):
        if query:
            st.info(f"Processing: {query}")
            result = handle_query(query, st.session_state.df)

            # Store in memory: query + result
            update_memory(query, result)

            if isinstance(result, tuple) and len(result) == 3:
                cleaned_df, results_df, task_type = result
                st.session_state.df = cleaned_df
                st.session_state.results_df = results_df
                st.session_state.task_type = task_type
                save_df(cleaned_df)
            elif isinstance(result, pd.DataFrame):
                st.session_state.df = result
                save_df(result)
                st.success("✅ Updated DataFrame:")
                st.dataframe(st.session_state.df.head())
            else:
                st.write(result)

    if st.button("Download Current CSV"):
        modified_csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=modified_csv, file_name="modified_data.csv", mime="text/csv")

    # Button to launch model comparison
    if st.session_state.results_df is not None and st.button("Compare Models"):
        from tasks.compare_models import explain_and_plot
        explain_and_plot(st.session_state.results_df, st.session_state.task_type)