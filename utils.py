
import requests

def mistral_generate(prompt):
    api_url = "https://api.mistral.agentspace.ai/generate"
    headers = {
        "Authorization": "your api key",
        "Content-Type": "application/json"
    }
    payload = {"model": "mistral-7b-instruct", "prompt": prompt, "max_tokens": 200}
    try:
        r = requests.post(api_url, headers=headers, json=payload)
        return r.json().get("text", "") if r.status_code == 200 else "# Error"
    except Exception as e:
        return f"# Mistral Exception: {str(e)}"

def huggingface_generate(prompt):
    try:
        headers = {"Authorization": "your api key"}
        payload = {"inputs": prompt}
        url = "https://api-inference.huggingface.co/models/bigcode/starcoder"
        response = requests.post(url, headers=headers, json=payload)
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"# HuggingFace Exception: {str(e)}"

def save_dataframe_as_csv(df):
    df.to_csv("updated_dataset.csv", index=False)
