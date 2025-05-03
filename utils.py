import pandas as pd
import numpy as np
from scipy.stats import zscore

def summarize_csv(df):
    summary = {
        'columns': list(df.columns),
        'missing_values': df.isnull().mean().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'sample': df.head(2).to_dict()
    }
    return summary

def safe_zscore(series):
    if pd.api.types.is_numeric_dtype(series):
        return zscore(series)
    else:
        return np.zeros(len(series))

def clean_generated_code(code):
    code = code.strip()
    if code.startswith("```") and code.endswith("```"):
        code = code[3:-3].strip()
        if code.startswith("python"):
            code = code[6:].strip()
    code = code.replace("Here's the code:", "").replace("```", "")
    return code