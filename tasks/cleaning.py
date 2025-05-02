from utils import summarize_csv, clean_generated_code
from llm_manager import ask_llm
import pandas as pd
import numpy as np

def clean_data(df):
    summary = summarize_csv(df)
    prompt = f"""
You are a Data Cleaning Expert.

TASK:
- Replace garbage values ('?', 'N/A', '', 'Unknown') with NaN.
- Fill numeric missing values with median, categorical with mode.
- Drop columns with >80% missing values.
- Remove outliers gently using 1.5*IQR method (but preserve maximum data).
- Prefer imputing values instead of dropping rows.

EXAMPLE:
import numpy as np
df.replace(['?', 'N/A', '', 'Unknown'], np.nan, inplace=True)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
df = df.loc[:, df.isnull().mean() < 0.8]
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= (Q1 - 1.5*IQR)) & (df[col] <= (Q3 + 1.5*IQR))]
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

DATASET SUMMARY:
Columns: {summary['columns']}
Missing: {summary['missing_values']}
Dtypes: {summary['dtypes']}
"""

    code = ask_llm(prompt)
    code = clean_generated_code(code)

    # ✅ Add syntax check
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as se:
        print("⚠️ Cleaning failed: LLM returned invalid code.")
        print(code)
        return df

    rows_before = df.shape[0]
    globals()['df'] = df.copy()

    exec(code, globals())

    df_cleaned = globals().get('df', None)

    if df_cleaned is None or not isinstance(df_cleaned, pd.DataFrame):
        print("⚠️ Cleaning failed. Reverting to original data.")
        return df  # rollback

    rows_after = df_cleaned.shape[0]

    if rows_after < 0.5 * rows_before:
        print("⚠️ Cleaning removed too many rows. Keeping original data.")
        return df
    else:
        return df_cleaned
