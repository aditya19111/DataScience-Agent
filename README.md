# Canny-Data Science bot
Smart CSV handling chatbot powered by Ollama + Streamlit + Langchain.

## Setup Instructions
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt
ollama pull llama3
streamlit run app.py
```

## Project Description
🧠 Agentic AI Chatbot for Data Science Automation

Agentic AI is an intelligent, conversational data science platform that turns natural language queries into automated data analysis, cleaning, and modeling workflows. Built with a modular, agent-based architecture, it empowers both technical and non-technical users to perform complex data science tasks simply by chatting with the system.

🚀 What It Does

Upload a dataset → Ask a question → Get insights.

The chatbot interprets user intent (e.g., “predict housing price”, “analyze correlations”, “clean missing data”), determines the right analytical path, and automatically executes the corresponding task — from data cleaning and exploratory analysis to model training and performance visualization.

⚙️ How It Works

User Interaction (Streamlit Frontend)
A clean, conversational interface lets users upload CSVs, enter queries, and instantly view results as data tables, metrics, and visual charts.

Intelligent Query Manager
Parses user input to detect intent and routes it to the appropriate agent:

clean_data() → Cleaning Agent

generate_auto_eda() → EDA Agent

train_model() → Model Agent

Cleaning Agent
Uses LLM-generated logic to clean and preprocess data safely, with validation and rollback mechanisms to prevent data corruption.

Model Training Module
Automatically detects task type — regression or classification — and applies the right algorithm (Linear, Logistic, or Random Forest).
Returns structured results (accuracy, RMSE, R², F1-score) for easy comparison.

Comparison Bot
Visualizes model performance in interactive charts and provides plain-language explanations of which model performs best and why.

Session & Memory Management
Keeps the conversation context alive, tracking recent queries and the current DataFrame to allow seamless multi-step workflows.

💡 Optional Extensions

FastAPI Integration – Expose /train, /predict, and /evaluate endpoints for external use.

AI Judge – Evaluate models based on statistical metrics and provide qualitative feedback.

AWS Deployment – Deployable via Lambda + API Gateway for cloud-based access.

🧩 Why It Stands Out

Modular, agentic architecture — easy to scale or add new agents

Secure execution of AI-generated cleaning and analysis code

Intelligent detection of regression vs. classification tasks

Visual, human-readable output for every operation

Real-time feedback and conversational ease for end-to-end automation

🏁 In a Sentence

Agentic AI transforms how we interact with data — turning complex machine learning workflows into natural, conversational experiences.
