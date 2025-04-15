
from query_handler import handle_query, handle_chart_query, handle_sql_query, handle_model_training
from utils import mistral_generate

def planner_agent(user_input):
    lowered = user_input.lower()

    if any(kw in lowered for kw in ["clean", "fix", "remove nulls", "handle missing", "dropna", "fill nulls", "remove duplicates"]):
        return "cleaning"

    elif any(kw in lowered for kw in ["plot", "histogram", "scatter", "bar chart", "distribution", "line chart", "visualize"]):
        return "chart"

    elif any(kw in lowered for kw in ["select", "from", "where", "join", "group by"]):
        return "sql"

    elif any(kw in lowered for kw in ["train", "accuracy", "classification", "regression", "model", "compare models"]):
        return "model"

    elif any(kw in lowered for kw in ["group", "filter", "sort", "aggregate", "mean", "average", "correlation", "transform", "rename", "drop column"]):
        return "query"

    else:
        return "text"

def executor_agent(intent, user_input, df):
    if intent == "query":
        return handle_query(user_input, df)
    elif intent == "chart":
        return handle_chart_query(user_input, df)
    elif intent == "sql":
        return handle_sql_query(user_input, df)
    elif intent == "model":
        return handle_model_training(df)
    elif intent == "cleaning":
        df_cleaned = df.copy()
        df_cleaned.drop_duplicates(inplace=True)
        df_cleaned.fillna(df_cleaned.mean(numeric_only=True), inplace=True)
        for col in df_cleaned.select_dtypes(include='object'):
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        return "✅ Data cleaned: nulls handled, duplicates removed.", df_cleaned, "text"
    else:
        return "Unable to understand your request.", df, "text"


def responder_agent(intent, result, output_type):
    if output_type == "text":
        return f"🧠 Result: {result}"
    elif output_type == "chart":
        return "📊 Here's your chart:"
    elif output_type == "table":
        return "📄 Here's the result table:"
    else:
        return result
