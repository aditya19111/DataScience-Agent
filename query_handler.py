
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import streamlit as st
from utils import mistral_generate, huggingface_generate
import pandasql as psql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def handle_query(query, df):
    prompt = f"Given a pandas DataFrame named df, write Python pandas code to: {query}."
    code = mistral_generate(prompt)
    if "Error" in code or "Exception" in code:
        code = huggingface_generate(prompt)
    try:
        local_vars = {'df': df.copy()}
        exec(code, {}, local_vars)
        df_updated = local_vars.get("df", df)
        return f"✅ Query executed.", df_updated, "text"
    except Exception as e:
        return f"❌ Error executing: {str(e)}", df, "text"

def handle_chart_query(query, df):
    prompt = f"Write a Python matplotlib/seaborn code snippet using DataFrame df to: {query}."
    code = mistral_generate(prompt)
    if "Error" in code or "Exception" in code:
        code = huggingface_generate(prompt)
    try:
        fig = plt.figure()
        local_vars = {'df': df, 'plt': plt, 'sns': sns}
        exec(code, {}, local_vars)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.image(buf)
        return "Chart rendered.", df, "chart"
    except Exception as e:
        return f"❌ Error drawing chart: {str(e)}", df, "text"

def handle_sql_query(query, df):
    try:
        result = psql.sqldf(query, {"df": df})
        st.dataframe(result)
        return "SQL query executed successfully.", df, "table"
    except Exception as e:
        return f"❌ SQL execution failed: {str(e)}", df, "text"

def handle_model_training(df):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import mean_squared_error, accuracy_score
    import pandas as pd

    df = df.dropna()
    df = df.select_dtypes(exclude=["object"])

    if df.shape[1] < 2 or df.shape[0] < 2:
        return "Dataset too small or not numeric enough.", df, "text"

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Detect regression vs classification
    is_classification = y.nunique() <= 20 and y.dtype in ['int', 'int64', 'int32']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []
    best_score = float('-inf')
    best_model_name = None

    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier()
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results.append((name, acc))
            if acc > best_score:
                best_score = acc
                best_model_name = name
        score_label = "Accuracy"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor()
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            score = -mse  # lower MSE = better
            results.append((name, mse))
            if score > best_score:
                best_score = score
                best_model_name = name
        score_label = "MSE (Lower is better)"

    st.subheader("📊 Model Performance Summary")
    st.dataframe(pd.DataFrame(results, columns=["Model", score_label]))
    return f"✅ Best model: {best_model_name} with {score_label} = {round(abs(best_score), 4)}", df, "text"

