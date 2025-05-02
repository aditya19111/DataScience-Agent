import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def explain_and_plot(results_df, task_type):
    st.header("🤖 Model Comparison Bot")

    if results_df is None or results_df.empty:
        st.warning("No model results to compare.")
        return

    st.subheader("🔍 Raw Evaluation Table")
    st.dataframe(results_df)

    # Determine which metrics to use
    if task_type == "classification":
        metrics = ["Accuracy", "F1 Score"]
    else:
        metrics = ["R2 Score", "RMSE"]

    # Plot each metric
    for metric in metrics:
        st.subheader(f"📊 {metric} Comparison")
        fig, ax = plt.subplots()
        bars = ax.bar(results_df["Model"], results_df[metric], color="skyblue")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Model")
        st.pyplot(fig)

        # Highlight best model
        if metric == "RMSE":
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        best_model = results_df.iloc[best_idx]["Model"]
        best_value = results_df.iloc[best_idx][metric]

        explanation = f"✅ **{best_model}** performed best in **{metric}** with a score of **{round(best_value, 4)}**."
        st.markdown(explanation)
