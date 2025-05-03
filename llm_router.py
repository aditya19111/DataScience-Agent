from langchain.llms import Ollama

llm = Ollama(model="llama3")

def route_query(user_query):
    routing_prompt = (
        "You are a smart routing agent in a data science chatbot.\n"
        "Your job is to read the user's query and decide which function or agent should handle it.\n\n"
        "Here are the available tools:\n"
        "- cleaning: For cleaning datasets — removing missing values, fixing formats, replacing garbage, etc.\n"
        "- eda: For visual and statistical data exploration (EDA)\n"
        "- train: For training machine learning models (classification/regression)\n"
        "- compare: For comparing trained model performances visually\n"
        "- unknown: If the intent is unclear or doesn't match any tool\n\n"
        "Examples:\n"
        "- clean the dataset → cleaning\n"
        "- remove nulls and outliers → cleaning\n"
        "- explore this data → eda\n"
        "- analyze trends → eda\n"
        "- train a model on age → train\n"
        "- predict income → train\n"
        "- predict City → train\n"
        "- can you predict City → train\n"
        "- show prediction for price → train\n"
        "- model on gender → train\n"
        "- compare models → compare\n"
        "- show me model results → compare\n"
        "- export csv → unknown\n"
        "- what's the weather → unknown\n\n"
        f"User Query:\n{user_query}\n\n"
        "Respond ONLY with one of: cleaning, eda, train, compare, unknown"
    )
    result = llm.invoke(routing_prompt).strip().lower()
    if result not in ["cleaning", "eda", "train", "compare"]:
        return "unknown"
    return result