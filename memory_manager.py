import streamlit as st

def update_memory(query, result=None):
    if 'memory' not in st.session_state:
        st.session_state.memory = []
    st.session_state.memory.append({'query': query, 'result': str(result)[:500]})
    if len(st.session_state.memory) > 5:
        st.session_state.memory.pop(0)

def get_memory_context():
    if 'memory' not in st.session_state or not st.session_state.memory:
        return ""
    context = "\n".join(
        f"User: {entry['query']}\nBot: {entry['result']}" for entry in st.session_state.memory
    )
    return f"Conversation history:\n{context}"