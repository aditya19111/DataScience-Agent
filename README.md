# Agentic AI v4 - Hugging Face Powered Conversational Data Scientist 🤖

This version removes Gemini completely and uses Hugging Face for all intent classification and natural language understanding.

---

## 🔄 Key Features

- Hugging Face model (`facebook/bart-large-mnli`) classifies user queries
- Auto-routes to the correct agent: cleaning, querying, charting, SQL, or modeling
- Mistral/Hugging Face LLMs generate Pandas, Matplotlib, or sklearn code
- Chat interface built with Streamlit
- Download cleaned dataset or visualize model performance
- No external Gemini setup required

---

## 🧠 How It Works

```text
User Query → Hugging Face Intent Classifier → Executes Matching Agent
```

---

## 🔐 API Keys Setup

Only 2 API keys are needed:

```env
MISTRAL_API_KEY=your_mistral_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

No need for GEMINI anymore.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run agentic_ai_app.py
```

---

## 📂 Files Overview

- `agentic_ai_app.py` – Main interface
- `agents.py` – Intent planner and executor logic
- `query_handler.py` – Task logic (chart, model, SQL, etc.)
- `utils.py` – Code generation via Mistral + Hugging Face
- `gemini/router.py` – Hugging Face-powered intent classification