from langchain.llms import Ollama
from memory_manager import get_memory_context

llm = Ollama(model="llama3")

def ask_llm(prompt):
    memory_context = get_memory_context()
    full_prompt = f"""STRICT INSTRUCTIONS:
- Output ONLY clean Python code.
- NO explanations or markdown.

{memory_context}

Current request:
{prompt}
"""
    return llm.invoke(full_prompt)