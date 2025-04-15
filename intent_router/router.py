from utils import mistral_generate
from agents import planner_agent
import json

def classify_user_intent(user_input, relevant_columns=None):
    try:
        prompt = (
            "You are a smart AI assistant that determines user intent and rewrites prompts for data analysis tasks.\n"
            f"User input: '{user_input}'\n"
            f"Relevant columns: {relevant_columns or []}\n"
            "Classify into one of: ['cleaning', 'chart', 'query', 'sql', 'model', 'text'].\n"
            "Return JSON: {\"intent\": str, \"rewritten_query\": str}"
        )

        response = mistral_generate(prompt)
        parsed = json.loads(response)

        intent = parsed.get("intent", None)
        rewritten_query = parsed.get("rewritten_query", user_input)

        if intent not in ["cleaning", "chart", "query", "sql", "model", "text"]:
            intent = planner_agent(rewritten_query)

        return {
            "intents": [intent],
            "rewritten_query": rewritten_query,
            "explanation": f"Mistral classified as '{intent}' and rewrote the query."
        }

    except Exception as e:
        fallback_intent = planner_agent(user_input)
        return {
            "intents": [fallback_intent],
            "rewritten_query": user_input,
            "explanation": f"Mistral failed. Fallback to planner_agent. Error: {str(e)}"
        }