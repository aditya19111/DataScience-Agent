from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_relevant_columns(user_prompt, columns, top_k=3):
    try:
        if not columns or len(columns) == 0:
            return []
        query_embedding = model.encode(user_prompt, convert_to_tensor=True)
        column_embeddings = model.encode(columns, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, column_embeddings)[0]
        top_indices = cosine_scores.argsort(descending=True)[:top_k]
        return [columns[i] for i in top_indices]
    except Exception:
        return []